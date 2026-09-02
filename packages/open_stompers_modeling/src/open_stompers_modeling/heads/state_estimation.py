"""State-estimation readout for heading-frame base motion."""

from math import isfinite

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import ArrayLike, Float

from .mlp import MlpHead
from .twist import TWIST_SIZE


@struct.dataclass
class StateEstimate:
    """Estimated heading-frame base motion.

    `velocity` follows `TWIST_COMPONENTS`: forward and lateral velocity in meters
    per second, then yaw rate in radians per second.
    """

    #: Realized twist shaped `(*batch, 3)`.
    velocity: jax.Array

    def displacement(self, control_interval: float) -> Float[jax.Array, "*batch 3"]:
        """Scale the estimated twist into a displacement over one control step.

        Args:
            control_interval: Positive step duration in seconds.

        Returns:
            Heading-frame forward displacement, lateral displacement, and yaw
            change over that step, in meters, meters, and radians.
        """
        interval = float(control_interval)
        if not isfinite(interval) or interval <= 0:
            raise ValueError("Control interval must be finite and positive")
        return self.velocity * jnp.asarray(interval, dtype=self.velocity.dtype)


class StateEstimationHead(nnx.Module):
    """Predict realized heading-frame base motion from model features.

    Set `action_size` to also condition on the residual action. That is the
    training-time movement estimator: its velocity times the control interval is
    the next-step forward, lateral, and yaw displacement.
    """

    def __init__(
        self,
        features: int,
        hidden_sizes: tuple[int, ...] = (),
        *,
        action_size: int = 0,
        rngs: nnx.Rngs,
    ) -> None:
        """Create the velocity readout.

        Args:
            features: Width of the incoming feature vector.
            hidden_sizes: Widths of optional ELU hidden layers before the readout.
            action_size: Residual-action width to concatenate, or zero to ignore it.
            rngs: Random streams for parameter initialization.
        """
        if action_size < 0:
            raise ValueError("Action width must be non-negative")
        self.features = features
        self.action_size = action_size
        self.readout = MlpHead(features + action_size, TWIST_SIZE, hidden_sizes, rngs=rngs)

    def __call__(
        self,
        features: Float[ArrayLike, "*batch features"],
        action: Float[ArrayLike, "*batch joints"] | None = None,
    ) -> StateEstimate:
        """Estimate the current heading-frame base twist.

        Args:
            features: Model features with optional leading batch axes.
            action: Residual action required when `action_size` is positive.

        Returns:
            Realized forward velocity, lateral velocity, and yaw rate.
        """
        values = _condition(self.features, self.action_size, features, action)
        return StateEstimate(self.readout(values))


def _condition(
    features: int,
    action_size: int,
    values: Float[ArrayLike, "*batch features"],
    action: Float[ArrayLike, "*batch joints"] | None,
) -> Float[jax.Array, "*batch conditioned"]:
    """Validate features and optionally append a residual action."""
    conditioned = jnp.asarray(values, dtype=jnp.float32)
    if conditioned.ndim < 1 or conditioned.shape[-1] != features:
        raise ValueError("Feature width does not match the head")
    if action_size == 0:
        if action is not None:
            raise ValueError("This state estimation head does not take an action")
        return conditioned
    if action is None:
        raise ValueError("Action-conditioned state estimation requires an action")
    residual = jnp.asarray(action, dtype=jnp.float32)
    if residual.shape[-1] != action_size or residual.shape[:-1] != conditioned.shape[:-1]:
        raise ValueError("Action shape does not match the head")
    return jnp.concatenate((conditioned, residual), axis=-1)
