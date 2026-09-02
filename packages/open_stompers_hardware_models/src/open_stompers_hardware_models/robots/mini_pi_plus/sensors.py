"""Mini Pi Plus sensors attached through link specs."""

from math import radians
from typing import Final

from open_stompers_hardware_models.robots.common import CameraSpec, ImuSpec

#: ZED Mini horizontal field of view.
ZED_MINI_HORIZONTAL_FOV: Final = radians(102.0)
ZED_MINI_IMAGE_WIDTH: Final = 2208
ZED_MINI_IMAGE_HEIGHT: Final = 1242

#: Base-aligned IMU. Matches HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic
#: ``pi_plus/mjcf/pi_20dof.xml`` at commit df3fcf16bf2cf645b8e3c24ee1e395d453f78f80.
BASE_LINK_IMU: Final = ImuSpec(name="imu")
HEAD_CAMERAS: Final[tuple[CameraSpec, ...]] = (
    CameraSpec(
        name="left_camera",
        fov=ZED_MINI_HORIZONTAL_FOV,
        width=ZED_MINI_IMAGE_WIDTH,
        height=ZED_MINI_IMAGE_HEIGHT,
        xyz=(0.0, 0.03, 0.01325),
        rpy=(-radians(90.0), 0.0, -radians(90.0)),
    ),
    CameraSpec(
        name="right_camera",
        fov=ZED_MINI_HORIZONTAL_FOV,
        width=ZED_MINI_IMAGE_WIDTH,
        height=ZED_MINI_IMAGE_HEIGHT,
        xyz=(0.0, -0.03, 0.01325),
        rpy=(-radians(90.0), 0.0, -radians(90.0)),
    ),
)
