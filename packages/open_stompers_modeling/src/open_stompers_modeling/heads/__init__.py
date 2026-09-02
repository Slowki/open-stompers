"""Common model output heads.

Heads read a feature vector and return a typed physical output. They share the
heading-frame twist layout `[forward_velocity, lateral_velocity, yaw_rate]`
where a command requests motion and a state estimate reports it. `MlpHead` is
the readout used by those heads and by further task-specific outputs.
"""

from .action import (
    DEFAULT_INITIAL_STANDARD_DEVIATION,
    LOG_STANDARD_DEVIATION_LIMITS,
    GaussianAction,
    GaussianActionHead,
)
from .mlp import MlpHead
from .state_estimation import StateEstimate, StateEstimationHead
from .twist import TWIST_COMPONENTS, TWIST_SIZE
from .velocity_command import VelocityCommand, VelocityCommandHead

__all__ = [
    "DEFAULT_INITIAL_STANDARD_DEVIATION",
    "LOG_STANDARD_DEVIATION_LIMITS",
    "TWIST_COMPONENTS",
    "TWIST_SIZE",
    "GaussianAction",
    "GaussianActionHead",
    "MlpHead",
    "StateEstimate",
    "StateEstimationHead",
    "VelocityCommand",
    "VelocityCommandHead",
]
