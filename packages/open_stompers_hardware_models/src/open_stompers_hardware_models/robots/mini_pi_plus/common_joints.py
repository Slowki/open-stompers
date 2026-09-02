"""Joint definitions shared by Mini Pi Plus SKUs."""

from typing import Final

from open_stompers_hardware_models.robots.common import JointSpec
from open_stompers_hardware_models.robots.mini_pi_plus.motors import (
    MOTOR_3536_32,
    MOTOR_4438_30,
    MOTOR_5047_36,
)

L_HIP_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="l_hip_pitch_joint",
    parent="base_link",
    child="l_hip_pitch_link",
    kind="revolute",
    xyz=(0.0, 0.0475, -0.0435),
    axis=(0.0, 1.0, 0.0),
    limits=(-3.12, 2.94),
    **MOTOR_5047_36,
)

L_HIP_ROLL_JOINT: Final[JointSpec] = JointSpec(
    name="l_hip_roll_joint",
    parent="l_hip_pitch_link",
    child="l_hip_roll_link",
    kind="revolute",
    xyz=(0.0, 0.034, 0.0),
    axis=(1.0, 0.0, 0.0),
    limits=(-0.17, 3.14),
    **MOTOR_5047_36,
)

L_CALF_JOINT: Final[JointSpec] = JointSpec(
    name="l_calf_joint",
    parent="l_thigh_link",
    child="l_calf_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.045),
    axis=(0.0, 1.0, 0.0),
    limits=(-2.14, 2.35),
    **MOTOR_5047_36,
)

L_ANKLE_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="l_ankle_pitch_joint",
    parent="l_calf_link",
    child="l_ankle_pitch_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.14),
    axis=(0.0, 1.0, 0.0),
    limits=(-0.97, 0.92),
    **MOTOR_5047_36,
)

L_ANKLE_ROLL_JOINT: Final[JointSpec] = JointSpec(
    name="l_ankle_roll_joint",
    parent="l_ankle_pitch_link",
    child="l_ankle_roll_link",
    kind="revolute",
    xyz=(0.051, 0.0, 0.0),
    axis=(1.0, 0.0, 0.0),
    limits=(-0.78, 0.78),
    **MOTOR_5047_36,
)

R_HIP_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="r_hip_pitch_joint",
    parent="base_link",
    child="r_hip_pitch_link",
    kind="revolute",
    xyz=(0.0, -0.0475, -0.0435),
    axis=(0.0, 1.0, 0.0),
    limits=(-3.12, 2.94),
    **MOTOR_5047_36,
)

R_HIP_ROLL_JOINT: Final[JointSpec] = JointSpec(
    name="r_hip_roll_joint",
    parent="r_hip_pitch_link",
    child="r_hip_roll_link",
    kind="revolute",
    xyz=(0.0, -0.034, 0.0),
    axis=(1.0, 0.0, 0.0),
    limits=(-3.14, 0.17),
    **MOTOR_5047_36,
)

R_CALF_JOINT: Final[JointSpec] = JointSpec(
    name="r_calf_joint",
    parent="r_thigh_link",
    child="r_calf_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.045),
    axis=(0.0, 1.0, 0.0),
    limits=(-2.14, 2.35),
    **MOTOR_5047_36,
)

R_ANKLE_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="r_ankle_pitch_joint",
    parent="r_calf_link",
    child="r_ankle_pitch_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.14),
    axis=(0.0, 1.0, 0.0),
    limits=(-0.97, 0.92),
    **MOTOR_5047_36,
)

R_ANKLE_ROLL_JOINT: Final[JointSpec] = JointSpec(
    name="r_ankle_roll_joint",
    parent="r_ankle_pitch_link",
    child="r_ankle_roll_link",
    kind="revolute",
    xyz=(0.051, 0.0, 0.0),
    axis=(1.0, 0.0, 0.0),
    limits=(-0.78, 0.78),
    **MOTOR_5047_36,
)

L_UPPER_ARM_JOINT: Final[JointSpec] = JointSpec(
    name="l_upper_arm_joint",
    parent="l_shoulder_roll_link",
    child="l_upper_arm_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.0724),
    axis=(0.0, 0.0, 1.0),
    limits=(-2.16, 2.16),
    **MOTOR_4438_30,
)

L_ELBOW_JOINT: Final[JointSpec] = JointSpec(
    name="l_elbow_joint",
    parent="l_upper_arm_link",
    child="l_elbow_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.038),
    axis=(0.0, 1.0, 0.0),
    limits=(-2.0, 2.0),
    **MOTOR_4438_30,
)

R_UPPER_ARM_JOINT: Final[JointSpec] = JointSpec(
    name="r_upper_arm_joint",
    parent="r_shoulder_roll_link",
    child="r_upper_arm_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.0724),
    axis=(0.0, 0.0, 1.0),
    limits=(-2.16, 2.16),
    **MOTOR_4438_30,
)

R_ELBOW_JOINT: Final[JointSpec] = JointSpec(
    name="r_elbow_joint",
    parent="r_upper_arm_link",
    child="r_elbow_link",
    kind="revolute",
    xyz=(0.0, 0.0, -0.038),
    axis=(0.0, 1.0, 0.0),
    limits=(-2.0, 2.0),
    **MOTOR_4438_30,
)

R_SHOULDER_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="r_shoulder_pitch_joint",
    parent="torso_link",
    child="r_shoulder_pitch_link",
    kind="revolute",
    xyz=(0.0, -0.1084, 0.117),
    axis=(0.0, 1.0, 0.0),
    limits=(-4.18, 1.04),
    **MOTOR_4438_30,
)

R_WRIST_JOINT: Final[JointSpec] = JointSpec(
    name="r_wrist_joint",
    parent="r_elbow_link",
    child="r_wrist_link",
    kind="fixed",
    xyz=(0.0, 0.0, -0.029),
)

L_SHOULDER_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="l_shoulder_pitch_joint",
    parent="torso_link",
    child="l_shoulder_pitch_link",
    kind="revolute",
    xyz=(0.0, 0.1084, 0.11698),
    axis=(0.0, 1.0, 0.0),
    limits=(-4.18, 1.04),
    **MOTOR_4438_30,
)

L_WRIST_JOINT: Final[JointSpec] = JointSpec(
    name="l_wrist_joint",
    parent="l_elbow_link",
    child="l_wrist_link",
    kind="fixed",
    xyz=(0.0, 0.0, -0.029),
)

HEAD_YAW_JOINT: Final[JointSpec] = JointSpec(
    name="head_yaw_joint",
    parent="torso_link",
    child="head_yaw_link",
    kind="revolute",
    xyz=(0.014, -0.00026667, 0.1981),
    axis=(0.0, 0.0, 1.0),
    limits=(-1.43, 1.43),
    **MOTOR_3536_32,
)

HEAD_PITCH_JOINT: Final[JointSpec] = JointSpec(
    name="head_pitch_joint",
    parent="head_yaw_link",
    child="head_pitch_link",
    kind="revolute",
    xyz=(0.0, 0.0, 0.0275),
    axis=(0.0, 1.0, 0.0),
    limits=(-1.23, 1.01),
    **MOTOR_3536_32,
)

CAMERA_JOINT: Final[JointSpec] = JointSpec(
    name="camera_joint",
    parent="head_pitch_link",
    child="camera_link",
    kind="fixed",
    xyz=(0.048925, 0.0053519, 0.058472),
)

R_THIGH_JOINT: Final[JointSpec] = JointSpec(
    name="r_thigh_joint",
    parent="r_hip_roll_link",
    child="r_thigh_link",
    kind="revolute",
    xyz=(-0.00035, 0.0, -0.0974),
    axis=(0.0, 0.0, 1.0),
    limits=(-2.87, 2.87),
    **MOTOR_5047_36,
)

L_THIGH_JOINT: Final[JointSpec] = JointSpec(
    name="l_thigh_joint",
    parent="l_hip_roll_link",
    child="l_thigh_link",
    kind="revolute",
    xyz=(-0.00035, 0.0, -0.0974),
    axis=(0.0, 0.0, 1.0),
    limits=(-2.87, 2.87),
    **MOTOR_5047_36,
)
