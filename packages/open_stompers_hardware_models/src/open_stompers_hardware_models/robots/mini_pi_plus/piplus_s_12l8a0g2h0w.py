"""PiPlus_S_12L8A0G2H0W (Mini Pi Plus S) model, described with LinkForge.

Configuration (from the variant code): 12 leg DoF, 8 arm DoF (wrists locked),
no gripper, 2 head DoF, no waist.

Transcribed from the upstream URDF in the `PiPlus_URDF` branch of
https://github.com/HighTorque-Robotics/HT_Robot_URDF (BSD-3-Clause), pinned to
commit c05257c64c79b624e773eae704752c09c6bdb007. Intentional deviations from the
upstream file:

* Mesh URIs are cache-relative instead of `package://` URIs.
* The upstream `<mujoco>` compiler tag has no LinkForge IR representation
  and is not reproduced on export.
"""

from pathlib import Path
from typing import Final

from linkforge.core import Robot

from open_stompers_hardware_models.robots.common import (
    CollisionSpec,
    JointSpec,
    LinkSpec,
    build_robot,
)
from open_stompers_hardware_models.robots.mini_pi_plus import assets as _assets
from open_stompers_hardware_models.robots.mini_pi_plus.common_joints import (
    CAMERA_JOINT,
    HEAD_PITCH_JOINT,
    HEAD_YAW_JOINT,
    L_ANKLE_PITCH_JOINT,
    L_ANKLE_ROLL_JOINT,
    L_CALF_JOINT,
    L_ELBOW_JOINT,
    L_SHOULDER_PITCH_JOINT,
    L_THIGH_JOINT,
    L_UPPER_ARM_JOINT,
    L_WRIST_JOINT,
    R_ANKLE_PITCH_JOINT,
    R_ANKLE_ROLL_JOINT,
    R_CALF_JOINT,
    R_ELBOW_JOINT,
    R_SHOULDER_PITCH_JOINT,
    R_THIGH_JOINT,
    R_UPPER_ARM_JOINT,
    R_WRIST_JOINT,
)
from open_stompers_hardware_models.robots.mini_pi_plus.materials import (
    MATERIAL_NAME,
    MATERIALS,
)
from open_stompers_hardware_models.robots.mini_pi_plus.motors import (
    MOTOR_4438_30,
    MOTOR_5047_36,
)
from open_stompers_hardware_models.robots.mini_pi_plus.sensors import BASE_LINK_IMU, HEAD_CAMERAS
from open_stompers_hardware_models.robots.mini_pi_plus.variants import MiniPiPlusVariant

VARIANT: Final[MiniPiPlusVariant] = (
    MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITHOUT_WAIST
)

ACTUATED_JOINTS: Final[tuple[str, ...]] = (
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
)

_LINKS: Final[tuple[LinkSpec, ...]] = (
    LinkSpec(
        name="base_link",
        mesh="base_link.STL",
        mass=1.49469794,
        com=(-0.00041617, 7.079e-05, -0.03929986),
        inertia=(0.00181946, 0.00111608, 0.00167502, -2.85e-06, 9.91e-06, 3.9e-07),
        collisions=(CollisionSpec(kind="box", dims=(0.05, 0.07, 0.05), xyz=(0.0, 0.0, -0.045)),),
        imus=(BASE_LINK_IMU,),
    ),
    LinkSpec(
        name="torso_link",
        mesh="torso_link.STL",
        mass=2.82493792,
        com=(0.00153764, -3.179e-05, 0.09595541),
        inertia=(0.01591654, 0.01340322, 0.01407319, 8.04e-06, -0.00011212, -2.99e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.155, 0.155, 0.155), xyz=(0.0, 0.0, 0.09)),),
    ),
    LinkSpec(
        name="r_shoulder_pitch_link",
        mesh="r_shoulder_pitch_link.STL",
        mass=0.13935188,
        com=(-0.0023165, -0.01720478, 0.00020825),
        inertia=(7.568e-05, 0.00012196, 0.00014519, -3.2e-06, 5e-08, -6.5e-07),
    ),
    LinkSpec(
        name="r_shoulder_roll_link",
        mesh="r_shoulder_roll_link.STL",
        mass=0.55615019,
        com=(-0.00164862, -0.00020989, -0.02488799),
        inertia=(0.00051303, 0.00052161, 0.0001534, -2e-07, 1.879e-05, -2.18e-06),
    ),
    LinkSpec(
        name="r_upper_arm_link",
        mesh="r_upper_arm_link.STL",
        mass=0.15318326,
        com=(3.107e-05, -0.00138646, -0.01213494),
        inertia=(0.00017707, 0.00011359, 0.00014214, -2.7e-07, 1.4e-07, -7.27e-06),
        collisions=(CollisionSpec(kind="cylinder", dims=(0.035, 0.095), xyz=(0.0, 0.0, 0.025)),),
    ),
    LinkSpec(
        name="r_elbow_link",
        mesh="r_elbow_link.STL",
        mass=0.31317542,
        com=(5.69e-05, -0.00308341, -0.00211549),
        inertia=(0.00010775, 9.336e-05, 9.783e-05, 5e-08, -3e-07, -4.5e-07),
    ),
    LinkSpec(
        name="r_wrist_link",
        mesh="r_wrist_link.STL",
        mass=0.07518056,
        com=(3.045e-05, 4.83e-06, -0.04078253),
        inertia=(0.00011809, 0.00011823, 1.105e-05, 0.0, 1.2e-07, 2e-08),
        collisions=(
            CollisionSpec(kind="cylinder", dims=(0.008, 0.03), xyz=(0.0, 0.0, -0.035)),
            CollisionSpec(kind="sphere", dims=(0.0175,), xyz=(0.0, 0.0, -0.095)),
        ),
    ),
    LinkSpec(
        name="l_shoulder_pitch_link",
        mesh="l_shoulder_pitch_link.STL",
        mass=0.13935188,
        com=(-0.00215543, 0.01720478, 0.00022511),
        inertia=(7.568e-05, 0.000122, 0.00014523, 3.32e-06, 1e-07, 6.7e-07),
    ),
    LinkSpec(
        name="l_shoulder_roll_link",
        mesh="l_shoulder_roll_link.STL",
        mass=0.55615015,
        com=(-0.00140359, 0.00022205, -0.02488799),
        inertia=(0.00051303, 0.00052202, 0.00015381, -1.08e-06, 2.293e-05, 8.2e-07),
    ),
    LinkSpec(
        name="l_upper_arm_link",
        mesh="l_upper_arm_link.STL",
        mass=0.15318327,
        com=(-0.00013535, 0.00129534, -0.01213424),
        inertia=(0.00017715, 0.00011361, 0.00014221, -1.2e-07, 5e-07, 7.46e-06),
        collisions=(CollisionSpec(kind="cylinder", dims=(0.035, 0.095), xyz=(0.0, 0.0, 0.025)),),
    ),
    LinkSpec(
        name="l_elbow_link",
        mesh="l_elbow_link.STL",
        mass=0.31317542,
        com=(-5.422e-05, 0.00312726, -0.00211549),
        inertia=(0.00010766, 9.336e-05, 9.774e-05, 5e-08, 2.9e-07, -2e-08),
    ),
    LinkSpec(
        name="l_wrist_link",
        mesh="l_wrist_link.STL",
        mass=0.07518056,
        com=(3.045e-05, 4.83e-06, -0.04078253),
        inertia=(0.00011809, 0.00011823, 1.105e-05, 0.0, 1.2e-07, 2e-08),
        collisions=(
            CollisionSpec(kind="cylinder", dims=(0.008, 0.03), xyz=(0.0, 0.0, -0.035)),
            CollisionSpec(kind="sphere", dims=(0.0175,), xyz=(0.0, 0.0, -0.095)),
        ),
    ),
    LinkSpec(
        name="head_yaw_link",
        mesh="head_yaw_link.STL",
        mass=0.20966327,
        com=(6.097e-05, 0.00133898, 0.02408163),
        inertia=(5.6e-05, 5.438e-05, 5.081e-05, 0.0, 2.4e-07, -3.2e-07),
        collisions=(CollisionSpec(kind="box", dims=(0.035, 0.04, 0.035), xyz=(0.0, 0.0, 0.0195)),),
    ),
    LinkSpec(
        name="head_pitch_link",
        mesh="head_pitch_link.STL",
        mass=0.33296951,
        com=(0.02665464, -0.00167129, 0.05552513),
        inertia=(0.00081268, 0.00055742, 0.00101639, 4.65e-06, -3.311e-05, -2.41e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.1, 0.11, 0.045), xyz=(0.026, 0.0, 0.055)),),
    ),
    LinkSpec(
        name="camera_link",
        mesh="camera_link.STL",
        mass=0.0584,
        com=(0.00963743, 0.00075253, 0.00058537),
        inertia=(7.259e-05, 5.53e-06, 7.249e-05, 9.8e-07, 0.0, 6e-08),
        cameras=HEAD_CAMERAS,
    ),
    LinkSpec(
        name="r_hip_pitch_link",
        mesh="r_hip_pitch_link.STL",
        mass=0.45931392,
        com=(7.92e-05, -0.03190261, 9.097e-05),
        inertia=(0.00027163, 0.00026494, 0.00027956, 1.14e-06, -6.6e-07, -4.1e-07),
    ),
    LinkSpec(
        name="r_hip_roll_link",
        mesh="r_hip_roll_link.STL",
        mass=0.55820116,
        com=(0.00102172, -0.00033797, -0.05845832),
        inertia=(0.00062111, 0.00074025, 0.00041298, -1.14e-06, -3.416e-05, -5.5e-07),
    ),
    LinkSpec(
        name="r_thigh_link",
        mesh="r_thigh_link.STL",
        mass=0.15484273,
        com=(0.00064557, -0.00075206, -0.02084055),
        inertia=(0.00018778, 0.00010058, 0.00016124, 4.5e-07, -8.8e-07, -5.6e-07),
        collisions=(CollisionSpec(kind="cylinder", dims=(0.035, 0.055), xyz=(0.0, 0.0, 0.065)),),
    ),
    LinkSpec(
        name="r_calf_link",
        mesh="r_calf_link.STL",
        mass=0.85852308,
        com=(-1.001e-05, -0.00237049, -0.06661243),
        inertia=(0.00416941, 0.00414652, 0.00042931, -6e-08, -9.9e-06, -5.94e-06),
        collisions=(
            CollisionSpec(kind="box", dims=(0.055, 0.055, 0.055), rpy=(1.57, 0.0, 0.0)),
            CollisionSpec(kind="box", dims=(0.01, 0.02, 0.055), xyz=(0.0, 0.0, -0.065)),
        ),
    ),
    LinkSpec(
        name="r_ankle_pitch_link",
        mesh="r_ankle_pitch_link.STL",
        mass=0.16352634,
        com=(0.00503044, 0.00051345, -0.00016685),
        inertia=(0.00011538, 0.00020824, 0.00028929, 4.2e-07, 1.65e-06, -6.3e-07),
        collisions=(CollisionSpec(kind="box", dims=(0.07, 0.07, 0.042), xyz=(0.0035, 0.0, 0.0)),),
    ),
    LinkSpec(
        name="r_ankle_roll_link",
        mesh="r_ankle_roll_link.STL",
        mass=0.61662552,
        com=(0.0046213, 8.211e-05, -0.01502099),
        inertia=(0.00054662, 0.00150792, 0.0013343, -8.4e-07, -0.00025529, 8e-07),
        collisions=(
            CollisionSpec(kind="box", dims=(0.06, 0.06, 0.042), xyz=(0.028, 0.0, 0.0)),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.13),
                xyz=(-0.017, -0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.164),
                xyz=(-0.0272, -0.009, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.164),
                xyz=(-0.0272, 0.009, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.13),
                xyz=(-0.017, 0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
        ),
    ),
    LinkSpec(
        name="l_hip_pitch_link",
        mesh="l_hip_pitch_link.STL",
        mass=0.45931808,
        com=(7.921e-05, 0.03190246, -0.0001562),
        inertia=(0.00027163, 0.00026494, 0.00027956, -1.14e-06, -3.9e-07, -2.25e-06),
    ),
    LinkSpec(
        name="l_hip_roll_link",
        mesh="l_hip_roll_link.STL",
        mass=0.55820115,
        com=(0.00087873, 1.341e-05, -0.05845832),
        inertia=(0.00062117, 0.00074046, 0.00041325, -1.07e-06, -3.583e-05, -1.6e-06),
    ),
    LinkSpec(
        name="l_thigh_link",
        mesh="l_thigh_link.STL",
        mass=0.15484236,
        com=(0.00071587, 0.00075215, -0.0208406),
        inertia=(0.00018778, 0.00010057, 0.00016122, -2.2e-07, -1.13e-06, 5.6e-07),
        collisions=(CollisionSpec(kind="cylinder", dims=(0.035, 0.055), xyz=(0.0, 0.0, 0.065)),),
    ),
    LinkSpec(
        name="l_calf_link",
        mesh="l_calf_link.STL",
        mass=0.85852288,
        com=(-1.001e-05, 0.00237048, -0.06661245),
        inertia=(0.00416941, 0.00414651, 0.00042931, 6e-08, 9.73e-06, 5.94e-06),
        collisions=(
            CollisionSpec(kind="box", dims=(0.055, 0.055, 0.055), rpy=(1.57, 0.0, 0.0)),
            CollisionSpec(kind="box", dims=(0.01, 0.02, 0.055), xyz=(0.0, 0.0, -0.065)),
        ),
    ),
    LinkSpec(
        name="l_ankle_pitch_link",
        mesh="l_ankle_pitch_link.STL",
        mass=0.16352634,
        com=(0.00503044, -0.00051345, -0.00016685),
        inertia=(0.00011538, 0.00020824, 0.00028929, -4.2e-07, 1.65e-06, 6.3e-07),
        collisions=(CollisionSpec(kind="box", dims=(0.07, 0.07, 0.042), xyz=(0.0035, 0.0, 0.0)),),
    ),
    LinkSpec(
        name="l_ankle_roll_link",
        mesh="l_ankle_roll_link.STL",
        mass=0.61662552,
        com=(0.00456214, 8.211e-05, -0.01256536),
        inertia=(0.00054662, 0.00150792, 0.0013343, -8.4e-07, -0.00025529, 8e-07),
        collisions=(
            CollisionSpec(kind="box", dims=(0.06, 0.06, 0.042), xyz=(0.028, 0.0, 0.0)),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.13),
                xyz=(-0.017, -0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.164),
                xyz=(-0.0272, -0.009, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.164),
                xyz=(-0.0272, 0.009, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.13),
                xyz=(-0.017, 0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
        ),
    ),
)

_JOINTS: Final[tuple[JointSpec, ...]] = (
    JointSpec(name="torso_joint", parent="base_link", child="torso_link", kind="fixed"),
    R_SHOULDER_PITCH_JOINT,
    JointSpec(
        name="r_shoulder_roll_joint",
        parent="r_shoulder_pitch_link",
        child="r_shoulder_roll_link",
        kind="revolute",
        xyz=(0.0, -0.038, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-3.34, 0.13),
        **MOTOR_4438_30,
    ),
    R_UPPER_ARM_JOINT,
    R_ELBOW_JOINT,
    R_WRIST_JOINT,
    L_SHOULDER_PITCH_JOINT,
    JointSpec(
        name="l_shoulder_roll_joint",
        parent="l_shoulder_pitch_link",
        child="l_shoulder_roll_link",
        kind="revolute",
        xyz=(0.0, 0.038, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-0.13, 3.34),
        **MOTOR_4438_30,
    ),
    L_UPPER_ARM_JOINT,
    L_ELBOW_JOINT,
    L_WRIST_JOINT,
    HEAD_YAW_JOINT,
    HEAD_PITCH_JOINT,
    CAMERA_JOINT,
    JointSpec(
        name="r_hip_pitch_joint",
        parent="base_link",
        child="r_hip_pitch_link",
        kind="revolute",
        xyz=(0.0, -0.0475, -0.0435),
        axis=(0.0, 1.0, 0.0),
        limits=(-1.76, 1.76),
        **MOTOR_5047_36,
    ),
    JointSpec(
        name="r_hip_roll_joint",
        parent="r_hip_pitch_link",
        child="r_hip_roll_link",
        kind="revolute",
        xyz=(0.0, -0.034, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-2.46, 0.17),
        **MOTOR_5047_36,
    ),
    R_THIGH_JOINT,
    R_CALF_JOINT,
    R_ANKLE_PITCH_JOINT,
    R_ANKLE_ROLL_JOINT,
    JointSpec(
        name="l_hip_pitch_joint",
        parent="base_link",
        child="l_hip_pitch_link",
        kind="revolute",
        xyz=(0.0, 0.0475, -0.0435),
        axis=(0.0, 1.0, 0.0),
        limits=(-1.76, 1.76),
        **MOTOR_5047_36,
    ),
    JointSpec(
        name="l_hip_roll_joint",
        parent="l_hip_pitch_link",
        child="l_hip_roll_link",
        kind="revolute",
        xyz=(0.0, 0.034, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-0.17, 2.46),
        **MOTOR_5047_36,
    ),
    L_THIGH_JOINT,
    L_CALF_JOINT,
    L_ANKLE_PITCH_JOINT,
    L_ANKLE_ROLL_JOINT,
)


def build() -> Robot:
    """Build the validated LinkForge model for this variant."""
    return build_robot(
        VARIANT,
        _LINKS,
        _JOINTS,
        mesh_resource=lambda filename: _assets.get_cache_relative_path(VARIANT, filename),
        materials=MATERIALS,
        default_material=MATERIAL_NAME,
    )


def ensure_meshes() -> list[Path]:
    """Download this variant's meshes into the local asset cache."""
    return _assets.ensure_meshes(VARIANT)
