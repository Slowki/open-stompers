"""Shared feature readout for output heads."""

from itertools import pairwise

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import ArrayLike, Float


class MlpHead(nnx.Module):
    """ELU feature trunk ending in a linear readout.

    With no hidden widths this is a single linear layer, matching the auxiliary
    readouts used on top of a transformer backbone.
    """

    def __init__(
        self,
        features: int,
        outputs: int,
        hidden_sizes: tuple[int, ...] = (),
        *,
        output_kernel_init: nnx.Initializer | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Create the readout layers.

        Args:
            features: Width of the incoming feature vector.
            outputs: Width of the readout.
            hidden_sizes: Widths of optional ELU hidden layers before the readout.
            output_kernel_init: Kernel initializer for the final linear layer.
            rngs: Random streams for parameter initialization.
        """
        if min((features, outputs, *hidden_sizes), default=0) < 1:
            raise ValueError("Head dimensions must be positive")
        self.features = features
        self.outputs = outputs
        self.hidden_sizes = hidden_sizes
        dimensions = (features, *hidden_sizes, outputs)
        self.layers = nnx.List(
            [
                nnx.Linear(
                    first,
                    second,
                    kernel_init=(
                        output_kernel_init
                        if output_kernel_init is not None and index == len(dimensions) - 2
                        else nnx.initializers.lecun_normal()
                    ),
                    precision=jax.lax.Precision.HIGHEST,
                    rngs=rngs,
                )
                for index, (first, second) in enumerate(pairwise(dimensions))
            ]
        )

    def __call__(
        self, features: Float[ArrayLike, "*batch features"]
    ) -> Float[jax.Array, "*batch outputs"]:
        """Project features to the readout width.

        Args:
            features: Model features with optional leading batch axes.

        Returns:
            The readout with the same leading batch axes.
        """
        hidden = jnp.asarray(features, dtype=jnp.float32)
        if hidden.ndim < 1 or hidden.shape[-1] != self.features:
            raise ValueError("Feature width does not match the head")
        for layer in self.layers[:-1]:
            hidden = jax.nn.elu(layer(hidden))
        return self.layers[-1](hidden)
