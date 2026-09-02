"""Neutral animation representation shared by generators, analysis, and exporters."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from open_stompers_datasets.rotations import base_velocities

#: Default sample interval of generated animations in seconds (50 Hz).
DEFAULT_SAMPLE_INTERVAL: Final = 0.02
#: Version of the on-disk animation schema.
SCHEMA_VERSION: Final = 1


class AnimationKind(StrEnum):
    """Whether an animation repeats or plays once."""

    LOOP = "loop"
    ACTION = "action"


class Gait(StrEnum):
    """Contact pattern family used to plan locomotion."""

    STAND = "stand"
    WALK = "walk"
    JOG = "jog"
    RUN = "run"


class Behavior(StrEnum):
    """Per-frame behavior label usable as a discrete policy input."""

    STAND = "stand"
    WALK = "walk"
    JOG = "jog"
    RUN = "run"
    SIT_DOWN = "sit_down"
    SIT = "sit"
    STAND_UP = "stand_up"
    JUMP = "jump"


@dataclass(frozen=True)
class LoopOffset:
    """Planar rigid transform from one loop cycle to the next, in the world frame."""

    #: Forward translation in meters.
    x: float = 0.0
    #: Lateral translation in meters.
    y: float = 0.0
    #: Heading change in radians.
    yaw: float = 0.0

    def power(self, cycles: int) -> "LoopOffset":
        """Compose this transform with itself a number of times.

        Args:
            cycles: Number of compositions, which may be negative.

        Returns:
            The transform after `cycles` loop repetitions.
        """
        step = self if cycles >= 0 else self.inverse()
        result = LoopOffset()
        for _ in range(abs(cycles)):
            result = result.compose(step)
        return result

    def compose(self, other: "LoopOffset") -> "LoopOffset":
        """Apply `other` after this transform."""
        cosine, sine = np.cos(other.yaw), np.sin(other.yaw)
        return LoopOffset(
            float(cosine * self.x - sine * self.y + other.x),
            float(sine * self.x + cosine * self.y + other.y),
            self.yaw + other.yaw,
        )

    def inverse(self) -> "LoopOffset":
        """Return the transform undoing this one."""
        cosine, sine = np.cos(self.yaw), np.sin(self.yaw)
        return LoopOffset(
            float(-(cosine * self.x + sine * self.y)),
            float(sine * self.x - cosine * self.y),
            -self.yaw,
        )

    def apply_positions(self, positions: NDArray) -> NDArray:
        """Transform world positions shaped (..., 3)."""
        cosine, sine = np.cos(self.yaw), np.sin(self.yaw)
        result = np.array(positions, dtype=np.float64, copy=True)
        result[..., 0] = cosine * positions[..., 0] - sine * positions[..., 1] + self.x
        result[..., 1] = sine * positions[..., 0] + cosine * positions[..., 1] + self.y
        return result

    def apply_quaternions(self, quaternions: NDArray) -> NDArray:
        """Rotate WXYZ world orientations shaped (..., 4) by the heading change."""
        half = self.yaw / 2
        rotation = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        scalar = rotation[0] * quaternions[..., :1] - rotation[3] * quaternions[..., 3:4]
        vector = (
            rotation[0] * quaternions[..., 1:]
            + quaternions[..., :1] * rotation[1:]
            + np.cross(rotation[1:], quaternions[..., 1:])
        )
        return np.concatenate((scalar, vector), axis=-1)


@dataclass(frozen=True)
class Animation:
    """Robot animation in meters, WXYZ quaternions, radians, and seconds.

    Loops hold exactly one cycle: the frame after the last one is the first frame
    transformed by `loop_offset`.
    """

    #: Unique animation name.
    name: str
    #: Hardware variant the animation was generated for.
    variant: str
    #: Loop or one-shot action.
    kind: AnimationKind
    #: Time between consecutive frames in seconds.
    sample_interval: float
    #: Joint names matching the joint columns.
    joint_names: tuple[str, ...]
    #: Robot root translations shaped (frames, 3).
    root_positions: NDArray[np.float32]
    #: Normalized root orientations shaped (frames, 4), in WXYZ order.
    root_quaternions: NDArray[np.float32]
    #: Joint angles shaped (frames, joints).
    joint_positions: NDArray[np.float32]
    #: Velocity command `[vx, vy, yaw_rate]` in the heading frame, shaped (frames, 3).
    commands: NDArray[np.float32]
    #: Behavior label of every frame.
    behaviors: tuple[Behavior, ...]
    #: Gait cycle phase in [0, 1) shaped (frames,), zero while not stepping.
    phases: NDArray[np.float32]
    #: Contact body names matching the contact columns.
    contact_names: tuple[str, ...]
    #: Planned ground contact flags shaped (frames, contacts).
    contacts: NDArray[np.bool_]
    #: Planned whole-body center of mass shaped (frames, 3).
    center_of_mass: NDArray[np.float32]
    #: Planned zero moment point on the ground shaped (frames, 2).
    zero_moment_point: NDArray[np.float32]
    #: Transform between loop cycles, identity for actions.
    loop_offset: LoopOffset = LoopOffset()
    #: Generator settings that produced the animation.
    parameters: dict[str, object] = field(default_factory=dict)
    #: Free-form provenance metadata.
    metadata: dict[str, object] = field(default_factory=dict)
    #: Joint velocities shaped (frames, joints), derived on construction.
    joint_velocities: NDArray[np.float32] = field(init=False)
    #: Base-frame linear velocities shaped (frames, 3), derived on construction.
    base_linear_velocities: NDArray[np.float32] = field(init=False)
    #: Base-frame angular velocities shaped (frames, 3), derived on construction.
    base_angular_velocities: NDArray[np.float32] = field(init=False)

    def __post_init__(self) -> None:
        """Validate arrays and derive velocities."""
        count = len(self.root_positions)
        if count < 2 or not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Animation must contain frames and unique joint names")
        if not np.isfinite(self.sample_interval) or self.sample_interval <= 0:
            raise ValueError("Sample interval must be finite and positive")
        for array, shape in (
            (self.root_positions, (count, 3)),
            (self.root_quaternions, (count, 4)),
            (self.joint_positions, (count, len(self.joint_names))),
            (self.commands, (count, 3)),
            (self.phases, (count,)),
            (self.center_of_mass, (count, 3)),
            (self.zero_moment_point, (count, 2)),
        ):
            if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Expected finite float32 animation array with shape {shape}")
        if self.contacts.shape != (count, len(self.contact_names)) or self.contacts.dtype != bool:
            raise ValueError("Contacts must be boolean with one column per contact name")
        if len(self.behaviors) != count:
            raise ValueError("Every frame needs a behavior label")
        if not np.allclose(np.linalg.norm(self.root_quaternions, axis=-1), 1, atol=1e-4):
            raise ValueError("Root quaternions must be normalized")
        positions, quaternions, joints = (
            self.root_positions,
            self.root_quaternions,
            self.joint_positions,
        )
        if self.kind is AnimationKind.LOOP:
            # Pad with neighbouring cycles so differences wrap smoothly.
            following, preceding = self.loop_offset, self.loop_offset.inverse()
            positions = np.concatenate(
                (
                    preceding.apply_positions(positions[-1:]),
                    positions,
                    following.apply_positions(positions[:1]),
                )
            ).astype(np.float32)
            quaternions = np.concatenate(
                (
                    preceding.apply_quaternions(quaternions[-1:]),
                    quaternions,
                    following.apply_quaternions(quaternions[:1]),
                )
            ).astype(np.float32)
            joints = np.concatenate((joints[-1:], joints, joints[:1]))
        linear, angular = base_velocities(positions, quaternions, self.sample_interval)
        velocities = np.gradient(joints, self.sample_interval, axis=0).astype(np.float32)
        if self.kind is AnimationKind.LOOP:
            linear, angular, velocities = linear[1:-1], angular[1:-1], velocities[1:-1]
        object.__setattr__(self, "base_linear_velocities", linear)
        object.__setattr__(self, "base_angular_velocities", angular)
        object.__setattr__(self, "joint_velocities", velocities)

    @property
    def frame_count(self) -> int:
        """Return the number of stored frames."""
        return len(self.root_positions)

    @property
    def duration(self) -> float:
        """Return the playback duration in seconds, one full cycle for loops."""
        frames = self.frame_count if self.kind is AnimationKind.LOOP else self.frame_count - 1
        return frames * self.sample_interval

    @property
    def times(self) -> NDArray[np.float64]:
        """Return the timestamp of every frame in seconds."""
        return np.arange(self.frame_count, dtype=np.float64) * self.sample_interval


__all__ = [
    "DEFAULT_SAMPLE_INTERVAL",
    "SCHEMA_VERSION",
    "Animation",
    "AnimationKind",
    "Behavior",
    "Gait",
    "LoopOffset",
]
