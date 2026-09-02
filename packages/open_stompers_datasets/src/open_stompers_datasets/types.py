"""Robot motion clip representation."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .rotations import base_velocities


@dataclass(frozen=True)
class MotionClip:
    """Robot motion in meters, WXYZ quaternions, radians, and seconds."""

    #: Robot root translations shaped (frames, 3).
    root_positions: NDArray[np.float32]
    #: Normalized root orientations shaped (frames, 4), in WXYZ order.
    root_quaternions: NDArray[np.float32]
    #: Base-frame linear velocities shaped (frames, 3), in meters per second.
    base_linear_velocities: NDArray[np.float32] = field(init=False)
    #: Base-frame angular velocities shaped (frames, 3), in radians per second.
    base_angular_velocities: NDArray[np.float32] = field(init=False)
    #: Robot joint angles shaped (frames, joints).
    joint_positions: NDArray[np.float32]
    #: Joint names matching the angle columns.
    joint_names: tuple[str, ...]
    #: Native time between consecutive samples in seconds.
    sample_interval: float
    #: Robot, source revision, motion identifier, and JSON-encoded ontology.
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        """Validate arrays, timing, and the joint layout."""
        count = len(self.root_positions)
        expected = (
            (self.root_positions, (count, 3)),
            (self.root_quaternions, (count, 4)),
            (self.joint_positions, (count, len(self.joint_names))),
        )
        if (
            count < 1
            or not self.joint_names
            or any(not isinstance(name, str) or not name for name in self.joint_names)
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("Motion must contain frames and unique joint names")
        if not np.isfinite(self.sample_interval) or self.sample_interval <= 0:
            raise ValueError("Sample interval must be finite and positive")
        for array, shape in expected:
            if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Expected finite float32 motion array with shape {shape}")
        if not np.allclose(np.linalg.norm(self.root_quaternions, axis=-1), 1, atol=1e-4):
            raise ValueError("Root quaternions must be normalized")
        linear, angular = base_velocities(
            self.root_positions, self.root_quaternions, self.sample_interval
        )
        object.__setattr__(self, "base_linear_velocities", linear)
        object.__setattr__(self, "base_angular_velocities", angular)

    @property
    def frame_count(self) -> int:
        """Return the number of motion samples."""
        return len(self.root_positions)
