"""Time-based sampling of animations, tiling loops across cycles."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from open_stompers_datasets.rotations import interpolate

from .types import Animation, AnimationKind, Behavior, LoopOffset


@dataclass(frozen=True)
class AnimationSample:
    """Animation state evaluated at arbitrary times."""

    #: Requested times in seconds shaped (samples,).
    times: NDArray[np.float64]
    #: Root translations shaped (samples, 3).
    root_positions: NDArray[np.float32]
    #: Root WXYZ orientations shaped (samples, 4).
    root_quaternions: NDArray[np.float32]
    #: Joint angles shaped (samples, joints).
    joint_positions: NDArray[np.float32]
    #: Velocity commands shaped (samples, 3).
    commands: NDArray[np.float32]
    #: Gait phases shaped (samples,).
    phases: NDArray[np.float32]
    #: Nearest-frame behavior labels.
    behaviors: tuple[Behavior, ...]
    #: Nearest-frame contact flags shaped (samples, contacts).
    contacts: NDArray[np.bool_]
    #: Planned centers of mass shaped (samples, 3).
    center_of_mass: NDArray[np.float32]
    #: Planned zero moment points shaped (samples, 2).
    zero_moment_point: NDArray[np.float32]


def _cycle_offsets(offset: LoopOffset, cycles: NDArray[np.int64]) -> dict[int, LoopOffset]:
    """Compute the transform of every distinct cycle index."""
    return {int(cycle): offset.power(int(cycle)) for cycle in np.unique(cycles)}


def _transform_rows(
    values: NDArray, cycles: NDArray[np.int64], offsets: dict[int, LoopOffset], quaternion: bool
) -> NDArray:
    """Apply the cycle transform of each row."""
    result = np.array(values, dtype=np.float64)
    for cycle, offset in offsets.items():
        rows = cycles == cycle
        result[rows] = (
            offset.apply_quaternions(values[rows])
            if quaternion
            else offset.apply_positions(values[rows])
        )
    return result


def sample_animation(animation: Animation, times: ArrayLike) -> AnimationSample:
    """Evaluate an animation at arbitrary times.

    Continuous channels are linearly interpolated, root orientations are spherically
    interpolated, and discrete channels use the nearest earlier frame. Loops repeat
    forever in both directions with each cycle moved by `loop_offset`; actions hold
    their first and last frames outside their duration.

    Args:
        animation: Animation to sample.
        times: Sample times in seconds.

    Returns:
        The sampled animation state.
    """
    requested = np.asarray(times, dtype=np.float64).reshape(-1)
    count, interval = animation.frame_count, animation.sample_interval
    position = requested / interval
    if animation.kind is AnimationKind.LOOP:
        lower = np.floor(position).astype(np.int64)
        fraction = position - lower
        upper = lower + 1
    else:
        position = np.clip(position, 0, count - 1)
        lower = np.minimum(np.floor(position).astype(np.int64), count - 2)
        fraction = position - lower
        upper = lower + 1
    lower_cycle, lower_index = np.divmod(lower, count)
    upper_cycle, upper_index = np.divmod(upper, count)
    offsets = _cycle_offsets(animation.loop_offset, np.concatenate((lower_cycle, upper_cycle)))
    column = fraction[:, None]

    def blend(values: NDArray) -> NDArray:
        return (1 - column) * values[lower_index] + column * values[upper_index]

    root_positions = (1 - column) * _transform_rows(
        animation.root_positions[lower_index], lower_cycle, offsets, quaternion=False
    ) + column * _transform_rows(
        animation.root_positions[upper_index], upper_cycle, offsets, quaternion=False
    )
    center_of_mass = (1 - column) * _transform_rows(
        animation.center_of_mass[lower_index], lower_cycle, offsets, quaternion=False
    ) + column * _transform_rows(
        animation.center_of_mass[upper_index], upper_cycle, offsets, quaternion=False
    )
    zero_moment_point = np.concatenate(
        (animation.zero_moment_point, np.zeros((count, 1), np.float32)), axis=1
    )
    zero_moment_point = (1 - column) * _transform_rows(
        zero_moment_point[lower_index], lower_cycle, offsets, quaternion=False
    ) + column * _transform_rows(
        zero_moment_point[upper_index], upper_cycle, offsets, quaternion=False
    )
    root_quaternions = interpolate(
        _transform_rows(
            animation.root_quaternions[lower_index], lower_cycle, offsets, quaternion=True
        ),
        _transform_rows(
            animation.root_quaternions[upper_index], upper_cycle, offsets, quaternion=True
        ),
        column,
    )
    lower_phase, upper_phase = animation.phases[lower_index], animation.phases[upper_index]
    # Unwrap the phase across the end of a gait cycle before blending.
    upper_phase = np.where(upper_phase < lower_phase - 0.5, upper_phase + 1, upper_phase)
    phases = np.mod((1 - fraction) * lower_phase + fraction * upper_phase, 1.0)
    nearest = np.where(fraction < 1 - 1e-9, lower_index, upper_index)
    return AnimationSample(
        times=requested,
        root_positions=root_positions.astype(np.float32),
        root_quaternions=root_quaternions.astype(np.float32),
        joint_positions=blend(animation.joint_positions).astype(np.float32),
        commands=blend(animation.commands).astype(np.float32),
        phases=phases.astype(np.float32),
        behaviors=tuple(animation.behaviors[index] for index in nearest),
        contacts=animation.contacts[nearest],
        center_of_mass=center_of_mass.astype(np.float32),
        zero_moment_point=zero_moment_point[:, :2].astype(np.float32),
    )


def resample(animation: Animation, sample_interval: float) -> Animation:
    """Resample an animation onto a new uniform frame interval.

    Args:
        animation: Animation to resample.
        sample_interval: New time between frames in seconds.

    Returns:
        The resampled animation.

    Raises:
        ValueError: If a loop cycle is not a whole number of new frames.
    """
    if animation.kind is AnimationKind.LOOP:
        frames = animation.duration / sample_interval
        if abs(frames - round(frames)) > 1e-6:
            raise ValueError("A loop cycle must contain a whole number of resampled frames")
        times = np.arange(round(frames)) * sample_interval
    else:
        times = np.arange(int(np.floor(animation.duration / sample_interval + 1e-9)) + 1)
        times = times * sample_interval
    sample = sample_animation(animation, times)
    return replace(
        animation,
        sample_interval=sample_interval,
        root_positions=sample.root_positions,
        root_quaternions=sample.root_quaternions,
        joint_positions=sample.joint_positions,
        commands=sample.commands,
        phases=sample.phases,
        behaviors=sample.behaviors,
        contacts=sample.contacts,
        center_of_mass=sample.center_of_mass,
        zero_moment_point=sample.zero_moment_point,
    )


__all__ = ["AnimationSample", "resample", "sample_animation"]
