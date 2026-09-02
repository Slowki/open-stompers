"""Dynamics-transformer history backbone without auxiliary prediction heads."""

from dataclasses import dataclass
from functools import cache
from typing import Final, Literal

import jax
import jax.numpy as jnp
import tokamax
from flax import nnx
from jaxtyping import Array, ArrayLike, Bool, Float


@dataclass(frozen=True)
class BackboneSettings:
    """Dynamics-transformer encoder dimensions."""

    token_size: int
    history: int = 50
    width: int = 128
    layers: int = 4
    heads: int = 4

    def __post_init__(self) -> None:
        """Reject incompatible attention dimensions."""
        if min(self.token_size, self.history, self.width, self.layers, self.heads) < 1:
            raise ValueError("Backbone dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Backbone width must be divisible by heads")


#: Lowest CUDA compute capability whose default tokamax attention kernels lower.
MINIMUM_DEFAULT_ATTENTION_CAPABILITY: Final = 8.9


@cache
def attention_implementation() -> Literal["xla"] | None:
    """Select tokamax's default attention kernels where they lower, and XLA elsewhere.

    Returns:
        `None` for tokamax's own selection on CPUs and recent GPUs, or `"xla"` on older GPUs,
        where its Triton kernel fails to lower.
    """
    device = jax.devices()[0]
    capability = getattr(device, "compute_capability", None)
    if device.platform != "gpu" or capability is None:
        return None
    return None if float(capability) >= MINIMUM_DEFAULT_ATTENTION_CAPABILITY else "xla"


class _Block(nnx.Module):
    """A pre-normalized causal transformer block."""

    def __init__(self, config: BackboneSettings, *, rngs: nnx.Rngs) -> None:
        """Create the block's normalization, attention, and MLP layers."""
        self.norm1 = nnx.LayerNorm(config.width, rngs=rngs)
        self.query = nnx.Linear(
            config.width, config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )
        self.key = nnx.Linear(
            config.width, config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )
        self.value = nnx.Linear(
            config.width, config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )
        self.attention_output = nnx.Linear(
            config.width, config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )
        self.portable_attention = False
        self.heads = config.heads
        self.head_size = config.width // config.heads
        self.norm2 = nnx.LayerNorm(config.width, rngs=rngs)
        self.hidden = nnx.Linear(
            config.width, 4 * config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )
        self.output = nnx.Linear(
            4 * config.width, config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )

    def __call__(
        self,
        x: Float[ArrayLike, "*batch time width"],
        mask: Bool[ArrayLike, "*batch heads time key_time"],
    ) -> Float[Array, "*batch time width"]:
        """Apply masked self-attention followed by a residual MLP."""
        x, mask = jnp.asarray(x), jnp.asarray(mask)
        normalized = self.norm1(x)
        query, key, value = (
            projection(normalized).reshape((*normalized.shape[:-1], self.heads, self.head_size))
            for projection in (self.query, self.key, self.value)
        )
        if self.portable_attention:
            scores = jnp.einsum(
                "...thd,...shd->...hts", query, key, precision=jax.lax.Precision.HIGHEST
            )
            scores /= jnp.sqrt(jnp.asarray(self.head_size, x.dtype))
            weights = jax.nn.softmax(jnp.where(mask, scores, jnp.finfo(x.dtype).min), axis=-1)
            attended = jnp.einsum(
                "...hts,...shd->...thd", weights, value, precision=jax.lax.Precision.HIGHEST
            )
        else:
            attended = tokamax.dot_product_attention(
                query,
                key,
                value,
                mask=mask,
                precision=jax.lax.Precision.HIGHEST,
                implementation=attention_implementation(),
            )
        x = x + self.attention_output(attended.reshape((*x.shape[:-1], -1)))
        return x + self.output(jax.nn.gelu(self.hidden(self.norm2(x))))


class DynamicsTransformerBackbone(nnx.Module):
    """Original dynamics-transformer causal encoder with an explicit sensor adapter."""

    def __init__(self, config: BackboneSettings, *, rngs: nnx.Rngs) -> None:
        """Restore the archived token projection, blocks and final normalization."""
        self.config = config
        self.token_projection = nnx.Linear(
            config.token_size, config.width, precision=jax.lax.Precision.HIGHEST, rngs=rngs
        )
        self.position = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (config.history, config.width))
        )
        self.blocks = nnx.List([_Block(config, rngs=rngs) for _ in range(config.layers)])
        self.final_norm = nnx.LayerNorm(config.width, rngs=rngs)

    def __call__(
        self,
        tokens: Float[ArrayLike, "*batch time features"],
        valid: Bool[ArrayLike, "*batch time"],
    ) -> Float[Array, "*batch time width"]:
        """Encode sensor/action history using the archived causal attention path."""
        tokens, valid = jnp.asarray(tokens), jnp.asarray(valid, bool)
        if tokens.shape[-2:] != (self.config.history, self.config.token_size):
            raise ValueError("History dimensions do not match backbone configuration")
        if valid.shape != tokens.shape[:-1]:
            raise ValueError("History mask shape does not match tokens")
        values = self.token_projection(jnp.where(valid[..., None], tokens, 0)) + self.position
        positions = jnp.arange(valid.shape[-1])
        causal = positions[None, :, None] >= positions[None, None, :]
        mask = (causal & valid[..., :, None] & valid[..., None, :])[..., None, :, :]
        for block in self.blocks:
            values = block(values, mask)
            values = jnp.where(valid[..., None], values, 0)
        return self.final_norm(values)
