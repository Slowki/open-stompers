"""Sensor projection parameters shared by model exporters and simulation."""

from math import atan, cos, degrees, sin, tan

from linkforge.core import Sensor


def camera_parameters(sensor: Sensor) -> tuple[tuple[float, ...], float]:
    """Compute MuJoCo image-right/up axes and vertical field of view in degrees."""
    if (camera := sensor.camera_info) is None:
        raise ValueError(f"Sensor {sensor.name!r} has no camera information")
    roll, pitch, yaw = sensor.origin.rpy
    cosine_roll, cosine_pitch, cosine_yaw = cos(roll), cos(pitch), cos(yaw)
    sine_roll, sine_pitch, sine_yaw = sin(roll), sin(pitch), sin(yaw)
    # Optical frames use +Z forward and +Y down; MuJoCo uses -Z forward and +Y up.
    axes = (
        cosine_yaw * cosine_pitch,
        sine_yaw * cosine_pitch,
        -sine_pitch,
        sine_yaw * cosine_roll - cosine_yaw * sine_pitch * sine_roll,
        -cosine_yaw * cosine_roll - sine_yaw * sine_pitch * sine_roll,
        -cosine_pitch * sine_roll,
    )
    vertical_fov = degrees(2 * atan(tan(camera.horizontal_fov / 2) * camera.height / camera.width))
    return axes, vertical_fov
