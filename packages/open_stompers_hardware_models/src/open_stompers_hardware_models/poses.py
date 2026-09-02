"""Nominal standing poses for hardware variants."""

from typing import Final

from open_stompers_hardware_models.robots.mini_pi_plus import (
    piplus_s_12l8a0g2h0w,
    piplus_s_12l8a0g2h1w,
    piplus_s_12l10a0g0h1w,
)
from open_stompers_hardware_models.variants import (
    HardwareVariant,
    MicroduckVariant,
    MiniPiPlusVariant,
)

#: Microduck crouched standing pose in radians, shared by every variant.
MICRODUCK_HOME_JOINT_ANGLE: Final[dict[str, float]] = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}

_MINI_PI_PLUS_JOINTS: Final[dict[MiniPiPlusVariant, tuple[str, ...]]] = {
    module.VARIANT: module.ACTUATED_JOINTS
    for module in (piplus_s_12l8a0g2h0w, piplus_s_12l8a0g2h1w, piplus_s_12l10a0g0h1w)
}


def _mini_pi_plus_home_joint_angle(name: str) -> float:
    """Return the Mini Pi Plus crouched home angle of one joint."""
    if "hip_pitch" in name or "ankle_pitch" in name:
        return -0.2
    if "calf" in name:
        return 0.4
    if name == "l_shoulder_roll_joint":
        return 0.12
    if name == "r_shoulder_roll_joint":
        return -0.12
    if "elbow" in name:
        return -0.3
    return 0.0


def home_pose(variant: HardwareVariant) -> dict[str, float]:
    """Return the crouched standing pose of every actuated joint.

    Args:
        variant: Hardware variant to describe.

    Returns:
        Joint angles in radians keyed by joint name.

    Raises:
        KeyError: If the hardware variant is unknown.
    """
    match variant:
        case MiniPiPlusVariant():
            return {
                name: _mini_pi_plus_home_joint_angle(name) for name in _MINI_PI_PLUS_JOINTS[variant]
            }
        case MicroduckVariant.ROLLER_SKATES:
            return MICRODUCK_HOME_JOINT_ANGLE | {
                "passive_LF_wheel": 0.0,
                "passive_LR_wheel": 0.0,
                "passive_RF_wheel": 0.0,
                "passive_RR_wheel": 0.0,
            }
        case MicroduckVariant():
            return dict(MICRODUCK_HOME_JOINT_ANGLE)
    raise KeyError(f"Unknown hardware variant '{variant}'")


__all__ = ["MICRODUCK_HOME_JOINT_ANGLE", "home_pose"]
