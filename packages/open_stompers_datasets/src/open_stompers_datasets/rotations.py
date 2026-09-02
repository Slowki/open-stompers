"""NumPy quaternion operations in WXYZ order."""

import numpy as np
from numpy.typing import NDArray


def multiply(left: NDArray, right: NDArray) -> NDArray:
    """Compose WXYZ quaternion rotations with broadcasting."""
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True
    )
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar, vector), axis=-1)


def conjugate(quaternion: NDArray) -> NDArray:
    """Invert a unit WXYZ quaternion."""
    return quaternion * np.asarray([1, -1, -1, -1])


def rotate(quaternion: NDArray, vector: NDArray) -> NDArray:
    """Rotate vectors by unit WXYZ quaternions."""
    imaginary = quaternion[..., 1:]
    first = 2 * np.cross(imaginary, vector)
    return vector + quaternion[..., :1] * first + np.cross(imaginary, first)


def interpolate(start: NDArray, end: NDArray, fraction: NDArray) -> NDArray:
    """Interpolate normalized WXYZ quaternions along their shortest arc."""
    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0, -end, end)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    denominator = np.maximum(sine, 1e-8)
    spherical = (
        np.sin((1 - fraction) * angle) / denominator * start
        + np.sin(fraction * angle) / denominator * end
    )
    linear = start + fraction * (end - start)
    result = np.where(dot > 0.9995, linear, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def base_velocities(
    positions: NDArray[np.float32],
    quaternions: NDArray[np.float32],
    sample_interval: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute linear and angular velocities in the base frame at every sample."""
    frame_count = len(positions)
    linear_world = np.zeros_like(positions)
    angular_world = np.zeros_like(positions)
    if frame_count > 1:
        linear_world[0] = (positions[1] - positions[0]) / sample_interval
        linear_world[-1] = (positions[-1] - positions[-2]) / sample_interval
        if frame_count > 2:
            linear_world[1:-1] = (positions[2:] - positions[:-2]) / (2 * sample_interval)
        # One-sided differences at the ends, central differences elsewhere.
        starts = np.concatenate(([0], np.arange(frame_count - 2), [frame_count - 2]))
        ends = np.concatenate(([1], np.arange(2, frame_count), [frame_count - 1]))
        elapsed = np.full(frame_count, 2 * sample_interval)
        elapsed[[0, -1]] = sample_interval
        angular_world = _angular_velocity(quaternions[starts], quaternions[ends], elapsed)
    conjugates = conjugate(quaternions)
    return (
        rotate(conjugates, linear_world).astype(np.float32),
        rotate(conjugates, angular_world).astype(np.float32),
    )


def _angular_velocity(
    start: NDArray[np.float32], end: NDArray[np.float32], elapsed: NDArray | float
) -> NDArray[np.float32]:
    """Compute shortest-arc world angular velocities between orientations."""
    relative = multiply(end, conjugate(start))
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    magnitude = np.linalg.norm(relative[..., 1:], axis=-1)
    angle = 2 * np.arctan2(magnitude, np.clip(relative[..., 0], -1, 1))
    scale = angle / (np.maximum(magnitude, 1e-30) * np.asarray(elapsed))
    velocity = relative[..., 1:] * scale[..., None]
    return np.where((magnitude < 1e-8)[..., None], 0.0, velocity).astype(np.float32)
