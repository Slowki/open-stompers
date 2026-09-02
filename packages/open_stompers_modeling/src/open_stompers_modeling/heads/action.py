"""Diagonal-Gaussian residual-action head."""

from math import isfinite
from typing import Final

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import ArrayLike, Float

from .mlp import MlpHead

#: Inclusive clip applied to log standard deviation when the head is evaluated.
LOG_STANDARD_DEVIATION_LIMITS: Final = (-4.0, 1.0)
#: Exploration scale used when a caller does not choose one.
DEFAULT_INITIAL_STANDARD_DEVIATION: Final = 0.5


@struct.dataclass
class GaussianAction:
    """Residual joint action from a diagonal Gaussian.

    Means are dimensionless residuals around the neutral pose. Deviations are
    positive and already clipped to the head's log-standard-deviation limits.
    """

    #: Action means shaped `(*batch, joints)`.
    mean: jax.Array
    #: Action standard deviations shaped `(*batch, joints)`.
    standard_deviation: jax.Array

    def log_probability(
        self, action: Float[ArrayLike, "*batch joints"]
    ) -> Float[jax.Array, "*batch"]:
        """Evaluate the log probability of residual actions.

        Args:
            action: Residual actions with the same shape as `mean`.

        Returns:
            Log probability of each action, reduced over joints.
        """
        sampled = jnp.asarray(action, dtype=self.mean.dtype)
        if sampled.shape != self.mean.shape:
            raise ValueError("Action shape does not match the Gaussian head")
        deviation = self.standard_deviation
        return jnp.sum(
            -0.5 * ((sampled - self.mean) / deviation) ** 2
            - jnp.log(deviation)
            - 0.5 * jnp.log(2 * jnp.pi),
            axis=-1,
        )


class GaussianActionHead(nnx.Module):
    """Gaussian residual-action readout.

    The mean kernel starts near zero so an untrained policy commands the neutral
    pose. The log standard deviation is a state-independent parameter.
    """

    def __init__(
        self,
        features: int,
        joints: int,
        hidden_sizes: tuple[int, ...] = (),
        *,
        initial_standard_deviation: float = DEFAULT_INITIAL_STANDARD_DEVIATION,
        rngs: nnx.Rngs,
    ) -> None:
        """Create the action mean readout and exploration parameter.

        Args:
            features: Width of the incoming feature vector.
            joints: Number of residual joint actions.
            hidden_sizes: Widths of optional ELU hidden layers before the mean.
            initial_standard_deviation: Positive initial exploration scale.
            rngs: Random streams for parameter initialization.
        """
        if not isfinite(initial_standard_deviation) or initial_standard_deviation <= 0:
            raise ValueError("Initial standard deviation must be finite and positive")
        self.joints = joints
        self.mean = MlpHead(
            features,
            joints,
            hidden_sizes,
            output_kernel_init=nnx.initializers.normal(0.01),
            rngs=rngs,
        )
        self.log_standard_deviation = nnx.Param(
            jnp.full((joints,), jnp.log(jnp.asarray(initial_standard_deviation, jnp.float32)))
        )

    def __call__(self, features: Float[ArrayLike, "*batch features"]) -> GaussianAction:
        """Infer residual-action means and clipped exploration deviations.

        Args:
            features: Model features with optional leading batch axes.

        Returns:
            A diagonal Gaussian over residual joint actions.
        """
        mean = self.mean(features)
        low, high = LOG_STANDARD_DEVIATION_LIMITS
        deviation = jnp.exp(jnp.clip(self.log_standard_deviation[...], low, high))
        return GaussianAction(mean, jnp.broadcast_to(deviation, mean.shape))
