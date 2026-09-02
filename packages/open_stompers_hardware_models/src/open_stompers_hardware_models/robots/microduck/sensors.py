"""MicroDuck measurement frames from the robot's Onshape export.

Matches pollen-robotics/microduck_rl robot_walk.xml at
3412ae50e5e49baf8c638c80734deb140e30101c. Positions are meters; quaternions are wxyz.
"""

from typing import Final

from open_stompers_hardware_models.mjcf_metadata import MjcfSensor, MjcfSite

SITES: Final = (
    MjcfSite(
        "left_foot", "ankle_left", (0.0, -0.0238146, -0.0140852), (0.0, 0.0, -0.707107, -0.707107)
    ),
    MjcfSite(
        "right_foot", "ankle_right", (0.0, -0.0238146, -0.0140852), (0.707107, -0.707107, 0.0, 0.0)
    ),
    MjcfSite(
        "mouth_tip", "jaw_soft", (-0.00809334, 0.0, -0.0777383), (0.704015, 0.0, 0.710185, 0.0)
    ),
)

SENSORS: Final = (
    MjcfSensor("orientation", "framequat", "imu", noise=0.001),
    MjcfSensor("angular-velocity", "gyro", "imu", noise=0.005),
    MjcfSensor("imu_ang_vel", "gyro", "imu"),
    MjcfSensor("imu_lin_vel", "velocimeter", "imu"),
    MjcfSensor("imu_accel", "accelerometer", "imu"),
    MjcfSensor("root_angmom", "subtreeangmom", "trunk_base"),
)
