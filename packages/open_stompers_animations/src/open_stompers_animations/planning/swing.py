"""Swing foot trajectories and smooth interpolation profiles."""

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Combined polynomial order of the clearance profile.
_CLEARANCE_ORDER: Final = 6


def minimum_jerk(fraction: ArrayLike) -> NDArray[np.float64]:
    """Map progress in [0, 1] onto a quintic with zero end velocity and acceleration."""
    value = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    return value**3 * (10 - 15 * value + 6 * value**2)


def clearance(fraction: ArrayLike, peak: float = 0.5) -> NDArray[np.float64]:
    """Return a unit-peak lift profile with zero end velocities.

    Args:
        fraction: Progress in [0, 1].
        peak: Progress at which the profile reaches one; earlier peaks lift faster.

    Returns:
        Profile values in [0, 1].
    """
    value = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    rising, falling = _CLEARANCE_ORDER * peak, _CLEARANCE_ORDER * (1 - peak)
    scale = peak**rising * (1 - peak) ** falling
    return value**rising * (1 - value) ** falling / scale


def wrap_angle(angle: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles into [-pi, pi)."""
    return np.mod(np.asarray(angle, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi


def swing_pose(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    height: float,
    fraction: ArrayLike,
    peak: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate a swing foot between two ground poses.

    Args:
        start: Liftoff pose `[x, y, yaw]`.
        end: Touchdown pose `[x, y, yaw]`.
        height: Peak clearance in meters.
        fraction: Swing progress in [0, 1] shaped (samples,).
        peak: Swing progress of the highest clearance.

    Returns:
        Sole positions shaped (samples, 3) and yaws shaped (samples,).
    """
    progress = minimum_jerk(fraction)[:, None]
    planar = start[:2] + progress * (end[:2] - start[:2])
    lift = height * clearance(fraction, peak)
    yaw = start[2] + progress[:, 0] * wrap_angle(end[2] - start[2])
    return np.concatenate((planar, lift[:, None]), axis=1), yaw


__all__ = ["clearance", "minimum_jerk", "swing_pose", "wrap_angle"]
