"""Velocity-command readout."""

import jax
from flax import nnx, struct
from jaxtyping import ArrayLike, Float

from .mlp import MlpHead
from .twist import TWIST_SIZE


@struct.dataclass
class VelocityCommand:
    """Desired heading-frame velocity command.

    `twist` follows `TWIST_COMPONENTS`: forward and lateral velocity in meters
    per second, then yaw rate in radians per second.
    """

    #: Desired twist shaped `(*batch, 3)`.
    twist: jax.Array


class VelocityCommandHead(nnx.Module):
    """Predict a desired heading-frame velocity command from model features.

    This is the planner-side counterpart of state estimation: it requests motion
    rather than reporting the motion the robot is already making.
    """

    def __init__(
        self,
        features: int,
        hidden_sizes: tuple[int, ...] = (),
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Create the command readout.

        Args:
            features: Width of the incoming feature vector.
            hidden_sizes: Widths of optional ELU hidden layers before the readout.
            rngs: Random streams for parameter initialization.
        """
        self.readout = MlpHead(features, TWIST_SIZE, hidden_sizes, rngs=rngs)

    def __call__(self, features: Float[ArrayLike, "*batch features"]) -> VelocityCommand:
        """Infer a desired heading-frame velocity command.

        Args:
            features: Model features with optional leading batch axes.

        Returns:
            Desired forward velocity, lateral velocity, and yaw rate.
        """
        return VelocityCommand(self.readout(features))
