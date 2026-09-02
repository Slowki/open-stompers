"""PiPlus_S_12L10A0G0H1W (Mini Pi Plus S) model, described with LinkForge.

Configuration (from the variant code): 12 leg DoF, 10 arm DoF including both
wrists, no gripper, fixed head, 1 waist DoF.

Transcribed from the upstream URDF in the `PiPlus_URDF` branch of
https://github.com/HighTorque-Robotics/HT_Robot_URDF (BSD-3-Clause), pinned to
commit c05257c64c79b624e773eae704752c09c6bdb007. Intentional deviations from the
upstream file:

* Robot name is the variant (upstream uses `pi_plus_23dof_251103`).
* Wrist joints are `continuous`; upstream declares `revolute` without
  limits, which is invalid URDF.
* Head yaw/pitch joints stay `fixed` exactly as declared upstream, even
  though its controller joint list still mentions them.
* Mesh URIs are cache-relative instead of URDF-relative paths.
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
    L_ANKLE_PITCH_JOINT,
    L_ANKLE_ROLL_JOINT,
    L_CALF_JOINT,
    L_ELBOW_JOINT,
    L_HIP_PITCH_JOINT,
    L_HIP_ROLL_JOINT,
    L_UPPER_ARM_JOINT,
    R_ANKLE_PITCH_JOINT,
    R_ANKLE_ROLL_JOINT,
    R_CALF_JOINT,
    R_ELBOW_JOINT,
    R_HIP_PITCH_JOINT,
    R_HIP_ROLL_JOINT,
    R_UPPER_ARM_JOINT,
)
from open_stompers_hardware_models.robots.mini_pi_plus.materials import (
    MATERIAL_NAME,
    MATERIALS,
)
from open_stompers_hardware_models.robots.mini_pi_plus.motors import (
    MOTOR_4438_30,
    MOTOR_5047_36,
)
from open_stompers_hardware_models.robots.mini_pi_plus.sensors import BASE_LINK_IMU
from open_stompers_hardware_models.robots.mini_pi_plus.variants import MiniPiPlusVariant

VARIANT: Final[MiniPiPlusVariant] = (
    MiniPiPlusVariant.TWELVE_LEG_TEN_ARM_WITHOUT_GRIPPERS_WITHOUT_HEAD_WITH_WAIST
)

ACTUATED_JOINTS: Final[tuple[str, ...]] = (
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "waist_yaw_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "l_wrist_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "r_wrist_joint",
)

_LINKS: Final[tuple[LinkSpec, ...]] = (
    LinkSpec(
        name="base_link",
        mesh="base_link.STL",
        mass=0.89256849,
        com=(-0.00045345, 0.0, -0.03862196),
        inertia=(0.00106054, 0.00041982, 0.00093191, -3.93e-06, 1.167e-05, 2e-08),
        collisions=(CollisionSpec(kind="box", dims=(0.05, 0.09, 0.05), xyz=(0.0, 0.0, -0.045)),),
        imus=(BASE_LINK_IMU,),
    ),
    LinkSpec(
        name="l_hip_pitch_link",
        mesh="l_hip_pitch_link.STL",
        mass=0.46752633,
        com=(0.00145357, 0.0319845, 0.00076488),
        inertia=(0.00018129, 0.00016324, 0.00017686, -2.76e-06, 1.25e-06, -2.1e-06),
        collisions=(
            CollisionSpec(
                kind="box", dims=(0.055, 0.055, 0.055), xyz=(0.0, 0.035, 0.0), rpy=(0.0, 1.57, 0.0)
            ),
        ),
    ),
    LinkSpec(
        name="l_hip_roll_link",
        mesh="l_hip_roll_link.STL",
        mass=0.57398951,
        com=(0.00077214, 0.00011543, -0.0588086),
        inertia=(0.00062136, 0.00074289, 0.00034162, -3.8e-07, -1.878e-05, 9.8e-07),
    ),
    LinkSpec(
        name="l_thigh_link",
        mesh="l_thigh_link.STL",
        mass=0.15007905,
        com=(0.00101286, -0.00069631, -0.0226902),
        inertia=(0.00016868, 9.384e-05, 0.00013886, -1e-07, -2.08e-06, -4.37e-06),
    ),
    LinkSpec(
        name="l_calf_link",
        mesh="l_calf_link.STL",
        mass=0.85789834,
        com=(0.00016713, 0.00269948, -0.06487937),
        inertia=(0.00409371, 0.00409958, 0.00023076, 1.52e-06, -2.679e-05, 2.22e-06),
        collisions=(
            CollisionSpec(kind="box", dims=(0.055, 0.055, 0.055), rpy=(1.57, 0.0, 0.0)),
            CollisionSpec(kind="box", dims=(0.01, 0.02, 0.055), xyz=(0.0, 0.0, -0.065)),
        ),
    ),
    LinkSpec(
        name="l_ankle_pitch_link",
        mesh="l_ankle_pitch_link.STL",
        mass=0.16558271,
        com=(0.00655369, -0.00218609, 1.446e-05),
        inertia=(0.00012024, 0.00018462, 0.00027198, -2.37e-06, -8e-08, 0.0),
        collisions=(CollisionSpec(kind="box", dims=(0.07, 0.07, 0.042), xyz=(0.0035, 0.0, 0.0)),),
    ),
    LinkSpec(
        name="l_ankle_roll_link",
        mesh="l_ankle_roll_link.STL",
        mass=0.51061803,
        com=(0.00374019, 0.00021427, -0.00921264),
        inertia=(0.00030023, 0.00110701, 0.00097968, -2.21e-06, -0.00016113, -4.1e-07),
        collisions=(
            CollisionSpec(kind="box", dims=(0.06, 0.06, 0.042), xyz=(0.028, 0.0, 0.0)),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.14),
                xyz=(-0.017, -0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.008, 0.16),
                xyz=(-0.024, -0.018, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.165),
                xyz=(-0.025, -0.01, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.169),
                xyz=(-0.0255, 0.0, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.165),
                xyz=(-0.025, 0.01, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.008, 0.16),
                xyz=(-0.024, 0.018, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.14),
                xyz=(-0.017, 0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
        ),
    ),
    LinkSpec(
        name="r_hip_pitch_link",
        mesh="r_hip_pitch_link.STL",
        mass=0.46752633,
        com=(0.00145354, -0.03244593, 0.00076792),
        inertia=(0.00018207, 0.00016325, 0.00017763, 2.47e-06, 1.32e-06, 9.3e-07),
        collisions=(
            CollisionSpec(
                kind="box", dims=(0.055, 0.055, 0.055), xyz=(0.0, -0.035, 0.0), rpy=(0.0, 1.57, 0.0)
            ),
        ),
    ),
    LinkSpec(
        name="r_hip_roll_link",
        mesh="r_hip_roll_link.STL",
        mass=0.57398951,
        com=(0.00077214, 4.589e-05, -0.0588086),
        inertia=(0.00062137, 0.00074289, 0.00034162, -4.2e-07, -1.878e-05, -4.1e-07),
    ),
    LinkSpec(
        name="r_thigh_link",
        mesh="r_thigh_link.STL",
        mass=0.15007905,
        com=(0.00106481, 0.00069567, -0.02268782),
        inertia=(0.00016869, 9.384e-05, 0.00013885, 1e-07, -2.01e-06, 4.37e-06),
    ),
    LinkSpec(
        name="r_calf_link",
        mesh="r_calf_link.STL",
        mass=0.85789835,
        com=(0.00031569, -0.00269949, -0.06487879),
        inertia=(0.00409368, 0.00409958, 0.00023079, -1.51e-06, 1.193e-05, -2.23e-06),
        collisions=(
            CollisionSpec(kind="box", dims=(0.055, 0.055, 0.055), rpy=(1.57, 0.0, 0.0)),
            CollisionSpec(kind="box", dims=(0.01, 0.02, 0.055), xyz=(0.0, 0.0, -0.065)),
        ),
    ),
    LinkSpec(
        name="r_ankle_pitch_link",
        mesh="r_ankle_pitch_link.STL",
        mass=0.16558271,
        com=(0.00655369, 0.00218609, -1.446e-05),
        inertia=(0.00012024, 0.00018462, 0.00027198, 2.37e-06, 8e-08, 0.0),
        collisions=(CollisionSpec(kind="box", dims=(0.07, 0.07, 0.042), xyz=(0.0035, 0.0, 0.0)),),
    ),
    LinkSpec(
        name="r_ankle_roll_link",
        mesh="r_ankle_roll_link.STL",
        mass=0.51061803,
        com=(0.00374019, 0.00021427, -0.00921264),
        inertia=(0.00030023, 0.00110701, 0.00097968, -2.21e-06, -0.00016113, -4.1e-07),
        collisions=(
            CollisionSpec(kind="box", dims=(0.06, 0.06, 0.042), xyz=(0.028, 0.0, 0.0)),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.14),
                xyz=(-0.017, -0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.008, 0.16),
                xyz=(-0.024, -0.018, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.165),
                xyz=(-0.025, -0.01, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.169),
                xyz=(-0.0255, 0.0, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.165),
                xyz=(-0.025, 0.01, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.008, 0.16),
                xyz=(-0.024, 0.018, -0.04),
                rpy=(0.0, -1.5708, 0.0),
            ),
            CollisionSpec(
                kind="cylinder",
                dims=(0.01, 0.14),
                xyz=(-0.017, 0.025, -0.04),
                rpy=(0.0, 1.5708, 0.0),
            ),
        ),
    ),
    LinkSpec(
        name="waist_yaw_link",
        mesh="waist_yaw_link.STL",
        mass=0.3940045,
        com=(-0.0002095, -0.00025712, -0.00446719),
        inertia=(0.00021467, 0.00022026, 0.00019691, 1.29e-06, 8.8e-07, 1.28e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.155, 0.155, 0.155), xyz=(0.0, 0.0, 0.12)),),
    ),
    LinkSpec(
        name="torso_link",
        mesh="torso_link.STL",
        mass=2.4997347,
        com=(0.00621927, 4.948e-05, 0.09663776),
        inertia=(0.01308492, 0.01114204, 0.01125017, 1.59e-05, 0.0003916, -6.29e-06),
    ),
    LinkSpec(
        name="l_shoulder_pitch_link",
        mesh="l_shoulder_pitch_link.STL",
        mass=0.14197719,
        com=(0.00035285, 0.01941039, 0.00013423),
        inertia=(7.492e-05, 0.00012391, 0.00014907, 6.8e-07, -1.5e-07, 4.2e-07),
    ),
    LinkSpec(
        name="l_shoulder_roll_link",
        mesh="l_shoulder_roll_link.STL",
        mass=0.49242354,
        com=(0.0013278, 0.00036622, -0.02242506),
        inertia=(0.00046209, 0.00045524, 0.00016271, 1.2e-06, -1.626e-05, 2.32e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.055, 0.055, 0.055), rpy=(0.0, 1.57, 0.0)),),
    ),
    LinkSpec(
        name="l_upper_arm_link",
        mesh="l_upper_arm_link.STL",
        mass=0.16207087,
        com=(-6.353e-05, 0.00043617, -0.01412292),
        inertia=(0.00019758, 0.00012003, 0.00014653, 1.6e-07, 1.6e-07, -7.4e-07),
    ),
    LinkSpec(
        name="l_elbow_link",
        mesh="l_elbow_link.STL",
        mass=0.28349125,
        com=(-0.00024727, 0.00242926, -0.00206736),
        inertia=(0.00010761, 0.00010511, 0.00010189, 5.3e-07, 2.1e-07, -5.6e-07),
    ),
    LinkSpec(
        name="l_wrist_link",
        mesh="l_wrist_link.STL",
        mass=0.07296157,
        com=(-1.174e-05, 2.952e-05, -0.04200021),
        inertia=(0.0001142, 0.00011418, 1.047e-05, 0.0, -4e-08, 1.1e-07),
        collisions=(
            CollisionSpec(kind="cylinder", dims=(0.008, 0.03), xyz=(0.0, 0.0, -0.035)),
            CollisionSpec(kind="sphere", dims=(0.0175,), xyz=(0.0, 0.0, -0.095)),
        ),
    ),
    LinkSpec(
        name="r_shoulder_pitch_link",
        mesh="r_shoulder_pitch_link.STL",
        mass=0.14197719,
        com=(0.00035285, -0.01941039, 0.00013423),
        inertia=(7.492e-05, 0.00012391, 0.00014907, -6.8e-07, -1.5e-07, -4.2e-07),
    ),
    LinkSpec(
        name="r_shoulder_roll_link",
        mesh="r_shoulder_roll_link.STL",
        mass=0.49240225,
        com=(0.00132741, -0.00037058, -0.02242698),
        inertia=(0.00046204, 0.00045519, 0.0001627, 3.3e-07, -1.626e-05, 5.74e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.055, 0.055, 0.055), rpy=(0.0, 1.57, 0.0)),),
    ),
    LinkSpec(
        name="r_upper_arm_link",
        mesh="r_upper_arm_link.STL",
        mass=0.16207087,
        com=(-6.353e-05, -0.00043617, -0.01412292),
        inertia=(0.00019758, 0.00012003, 0.00014653, -1.6e-07, 1.6e-07, 7.4e-07),
    ),
    LinkSpec(
        name="r_elbow_link",
        mesh="r_elbow_link.STL",
        mass=0.28349125,
        com=(0.00024727, -0.00242926, -0.00206736),
        inertia=(0.00010761, 0.00010511, 0.00010189, 5.3e-07, -2.1e-07, 5.6e-07),
    ),
    LinkSpec(
        name="r_wrist_link",
        mesh="r_wrist_link.STL",
        mass=0.07296157,
        com=(-3.138e-05, -4.98e-06, -0.04200021),
        inertia=(0.00011418, 0.0001142, 1.047e-05, 0.0, -1.2e-07, -2e-08),
        collisions=(
            CollisionSpec(kind="cylinder", dims=(0.008, 0.03), xyz=(0.0, 0.0, -0.035)),
            CollisionSpec(kind="sphere", dims=(0.0175,), xyz=(0.0, 0.0, -0.095)),
        ),
    ),
    LinkSpec(
        name="head_yaw_link",
        mesh="head_yaw_link.STL",
        mass=0.18404596,
        com=(-8.774e-05, 0.0001841, 0.02306327),
        inertia=(3.968e-05, 3.515e-05, 3.343e-05, 3.9e-07, 7e-08, -2.27e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.035, 0.04, 0.035), xyz=(0.0, 0.0, 0.0195)),),
    ),
    LinkSpec(
        name="head_pitch_link",
        mesh="head_pitch_link.STL",
        mass=0.48381331,
        com=(0.03264143, -0.0021897, 0.05059467),
        inertia=(0.0010631, 0.00108641, 0.00180811, 2.764e-05, 3.82e-06, 7.06e-06),
        collisions=(CollisionSpec(kind="box", dims=(0.13, 0.16, 0.055), xyz=(0.025, 0.0, 0.05)),),
    ),
)

_JOINTS: Final[tuple[JointSpec, ...]] = (
    L_HIP_PITCH_JOINT,
    L_HIP_ROLL_JOINT,
    JointSpec(
        name="l_thigh_joint",
        parent="l_hip_roll_link",
        child="l_thigh_link",
        kind="revolute",
        xyz=(-0.00025, 0.0, -0.0977),
        axis=(0.0, 0.0, 1.0),
        limits=(-2.87, 2.87),
        **MOTOR_5047_36,
    ),
    L_CALF_JOINT,
    L_ANKLE_PITCH_JOINT,
    L_ANKLE_ROLL_JOINT,
    R_HIP_PITCH_JOINT,
    R_HIP_ROLL_JOINT,
    JointSpec(
        name="r_thigh_joint",
        parent="r_hip_roll_link",
        child="r_thigh_link",
        kind="revolute",
        xyz=(-0.00025, 0.0, -0.0977),
        axis=(0.0, 0.0, 1.0),
        limits=(-2.87, 2.87),
        **MOTOR_5047_36,
    ),
    R_CALF_JOINT,
    R_ANKLE_PITCH_JOINT,
    R_ANKLE_ROLL_JOINT,
    JointSpec(
        name="waist_yaw_joint",
        parent="base_link",
        child="waist_yaw_link",
        kind="revolute",
        xyz=(0.0, 0.0, 0.02935),
        axis=(0.0, 0.0, 1.0),
        limits=(-3.14, 3.14),
        **MOTOR_5047_36,
    ),
    JointSpec(
        name="torso_joint",
        parent="waist_yaw_link",
        child="torso_link",
        kind="fixed",
        xyz=(0.0, 0.0, 0.02935),
    ),
    JointSpec(
        name="l_shoulder_pitch_joint",
        parent="torso_link",
        child="l_shoulder_pitch_link",
        kind="revolute",
        xyz=(0.0, 0.1069, 0.117),
        axis=(0.0, 1.0, 0.0),
        limits=(-4.18, 1.04),
        **MOTOR_4438_30,
    ),
    JointSpec(
        name="l_shoulder_roll_joint",
        parent="l_shoulder_pitch_link",
        child="l_shoulder_roll_link",
        kind="revolute",
        xyz=(0.0, 0.038, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-0.13, 3.0),
        **MOTOR_4438_30,
    ),
    L_UPPER_ARM_JOINT,
    L_ELBOW_JOINT,
    JointSpec(
        name="l_wrist_joint",
        parent="l_elbow_link",
        child="l_wrist_link",
        kind="continuous",
        xyz=(0.0, 0.0, -0.029),
        axis=(0.0, 0.0, 1.0),
    ),
    JointSpec(
        name="r_shoulder_pitch_joint",
        parent="torso_link",
        child="r_shoulder_pitch_link",
        kind="revolute",
        xyz=(0.0, -0.1069, 0.117),
        axis=(0.0, 1.0, 0.0),
        limits=(-4.18, 1.04),
        **MOTOR_4438_30,
    ),
    JointSpec(
        name="r_shoulder_roll_joint",
        parent="r_shoulder_pitch_link",
        child="r_shoulder_roll_link",
        kind="revolute",
        xyz=(0.0, -0.038, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-3.0, 0.13),
        **MOTOR_4438_30,
    ),
    R_UPPER_ARM_JOINT,
    R_ELBOW_JOINT,
    JointSpec(
        name="r_wrist_joint",
        parent="r_elbow_link",
        child="r_wrist_link",
        kind="continuous",
        xyz=(0.0, 0.0, -0.029),
        axis=(0.0, 0.0, 1.0),
    ),
    JointSpec(
        name="head_yaw_joint",
        parent="torso_link",
        child="head_yaw_link",
        kind="fixed",
        xyz=(0.0, 0.00025, 0.1981),
    ),
    JointSpec(
        name="head_pitch_joint",
        parent="head_yaw_link",
        child="head_pitch_link",
        kind="fixed",
        xyz=(0.0, 0.0, 0.0255),
    ),
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
