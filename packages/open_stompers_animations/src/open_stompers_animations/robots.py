"""Per-robot animation settings."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from open_stompers_hardware_models import HardwareVariant, MicroduckVariant, MiniPiPlusVariant

from .types import Gait


@dataclass(frozen=True)
class SupportPad:
    """Box under the trunk that carries the robot while sitting."""

    #: Body the pad is attached to.
    body: str
    #: Pad center in the body frame, in meters.
    position: tuple[float, float, float]
    #: Pad half extents, in meters.
    half_size: tuple[float, float, float]


@dataclass(frozen=True)
class GaitSettings:
    """Default timing and shape of one gait."""

    #: Fraction of a gait cycle each foot spends on the ground.
    duty_factor: float
    #: Peak swing foot clearance in meters.
    step_height: float
    #: Forward speed used by actions that do not specify one, in meters per second.
    default_speed: float
    #: Steps per minute, one step being one touchdown; unset uses the human cadence at
    #: the dynamically similar speed, scaled by the robot height.
    cadence: float | None = None
    #: Least knee bend of loaded legs in radians; smaller is straighter.
    stance_knee_flexion: float = 0.2
    #: Forward lean of the base in radians.
    base_pitch: float = 0.0
    #: Final fraction of each stance spent rolling onto the toe.
    heel_off: float = 0.0
    #: Toe-down foot pitch reached at liftoff, in radians.
    toe_off_pitch: float = 0.0
    #: Swing progress of the highest foot clearance.
    swing_peak: float = 0.5
    #: Elbow bend in radians, the home pose when unset.
    elbow_flexion: float | None = None
    #: Forward offset of the arm swing center in radians, shifting mass forward.
    arm_forward_bias: float = 0.0
    #: Largest arm swing either way from the swing center, in radians.
    arm_swing_limit: float = 0.35
    #: Lateral distance between the foot centers in meters, the standing width when unset.
    step_width: float | None = None


@dataclass(frozen=True)
class JumpSettings:
    """Vertical jump geometry."""

    #: Center of mass drop before pushing off, in meters.
    crouch_depth: float
    #: Center of mass rise above standing at takeoff, in meters.
    takeoff_extension: float
    #: Default apex height of the center of mass above takeoff, in meters.
    height: float


@dataclass(frozen=True)
class AnimationRobot:
    """Kinematic roles and motion limits of one hardware variant."""

    #: Hardware variant described by these settings.
    variant: HardwareVariant
    #: Left and right foot bodies.
    feet: tuple[str, str]
    #: Joints solved by inverse kinematics.
    leg_joints: tuple[str, ...]
    #: Knee joints and the sign of their flexion direction.
    knee_flexion: Mapping[str, float]
    #: Left and right joints whose anchors act as hip pivots.
    hip_joints: tuple[str, str]
    #: Left and right joints whose anchors act as ankle pivots.
    ankle_joints: tuple[str, str]
    #: Left and right joints that swing the arms, if the robot has arms.
    arm_swing_joints: tuple[str, str] | None
    #: Elbow joints and the sign of their flexion direction.
    elbow_flexion_signs: Mapping[str, float]
    #: Pad used for supported sitting.
    support_pad: SupportPad
    #: Arm swing in radians per meter of foot travel relative to the body.
    arm_swing_gain: float
    #: Largest travel of a swing foot between placements, in meters.
    maximum_foot_travel: float
    #: Largest distance a stance foot sweeps under the body, in meters.
    maximum_stance_sweep: float
    #: Smallest lateral distance between foot centers, in meters.
    minimum_foot_separation: float
    #: Largest yaw difference between the feet, in radians.
    maximum_foot_yaw_difference: float
    #: Lateral shift of the single support zero moment point toward the body, in meters.
    support_inset: float
    #: Base orientation weight scales about the base roll, pitch, and yaw axes.
    base_orientation_axes: tuple[float, float, float]
    #: Largest accepted stance foot or center of mass tracking error, in meters.
    position_tolerance: float
    #: Largest accepted swing foot tracking error, in meters.
    swing_tolerance: float
    #: Horizontal distance of the pad center behind the feet when seated, in meters.
    sit_back_distance: float
    #: Default sideways stepping speed in meters per second.
    side_step_speed: float
    #: Vertical jump settings.
    jump: JumpSettings
    #: Default settings of every moving gait.
    gaits: Mapping[Gait, GaitSettings] = field(default_factory=dict)

    @property
    def contact_names(self) -> tuple[str, str, str]:
        """Return the contact channel names used in animations."""
        return ("left_foot", "right_foot", "support_pad")


_MINI_PI_PLUS_LEGS: Final = tuple(
    f"{side}_{joint}_joint"
    for side in ("l", "r")
    for joint in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")
)

_MICRODUCK_LEGS: Final = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
)


def _mini_pi_plus(variant: MiniPiPlusVariant) -> AnimationRobot:
    """Describe a Mini Pi Plus variant."""
    return AnimationRobot(
        variant=variant,
        feet=("l_ankle_roll_link", "r_ankle_roll_link"),
        leg_joints=_MINI_PI_PLUS_LEGS,
        knee_flexion={"l_calf_joint": 1.0, "r_calf_joint": 1.0},
        hip_joints=("l_hip_roll_joint", "r_hip_roll_joint"),
        ankle_joints=("l_ankle_pitch_joint", "r_ankle_pitch_joint"),
        arm_swing_joints=("l_shoulder_pitch_joint", "r_shoulder_pitch_joint"),
        elbow_flexion_signs={"l_elbow_joint": -1.0, "r_elbow_joint": -1.0},
        # Matches the supported-sit pad of the interactive simulation scene.
        support_pad=SupportPad("base_link", (0.0, 0.0, -0.1301), (0.055, 0.055, 0.005)),
        arm_swing_gain=2.0,
        maximum_foot_travel=0.5,
        maximum_stance_sweep=0.2,
        minimum_foot_separation=0.08,
        maximum_foot_yaw_difference=0.6,
        support_inset=0.015,
        base_orientation_axes=(1.0, 1.0, 1.0),
        position_tolerance=0.007,
        swing_tolerance=0.04,
        sit_back_distance=0.16,
        side_step_speed=0.08,
        jump=JumpSettings(crouch_depth=0.06, takeoff_extension=0.004, height=0.04),
        gaits={
            Gait.WALK: GaitSettings(
                duty_factor=0.6,
                step_height=0.04,
                default_speed=0.25,
                stance_knee_flexion=0.2,
                heel_off=0.25,
                toe_off_pitch=0.3,
            ),
            Gait.JOG: GaitSettings(
                duty_factor=0.45,
                step_height=0.05,
                default_speed=0.4,
                stance_knee_flexion=0.3,
                base_pitch=0.12,
                heel_off=0.35,
                toe_off_pitch=0.5,
                elbow_flexion=1.2,
                arm_forward_bias=0.1,
                arm_swing_limit=0.5,
            ),
            Gait.RUN: GaitSettings(
                duty_factor=0.38,
                step_height=0.06,
                default_speed=0.6,
                stance_knee_flexion=0.35,
                base_pitch=0.2,
                heel_off=0.35,
                toe_off_pitch=0.5,
                elbow_flexion=1.4,
                arm_forward_bias=0.15,
                arm_swing_limit=0.6,
            ),
        },
    )


def _microduck(variant: MicroduckVariant) -> AnimationRobot:
    """Describe the legged Microduck."""
    return AnimationRobot(
        variant=variant,
        feet=("ankle_left", "ankle_right"),
        leg_joints=_MICRODUCK_LEGS,
        knee_flexion={"left_knee": 1.0, "right_knee": -1.0},
        hip_joints=("left_hip_pitch", "right_hip_pitch"),
        ankle_joints=("left_ankle", "right_ankle"),
        arm_swing_joints=None,
        elbow_flexion_signs={},
        # Matches the supported-sit pad of the interactive simulation scene.
        support_pad=SupportPad("trunk_base", (0.0, 0.0, -0.0551), (0.04, 0.04, 0.005)),
        arm_swing_gain=0.0,
        maximum_foot_travel=0.13,
        maximum_stance_sweep=0.06,
        minimum_foot_separation=0.045,
        maximum_foot_yaw_difference=0.4,
        support_inset=0.016,
        # Without ankle roll joints the trunk must roll to sway over the stance foot, and
        # each foot only stays flat under its own hip, so gaits keep the standing width.
        base_orientation_axes=(0.01, 1.0, 1.0),
        # Five-joint legs cannot match every foot orientation exactly.
        position_tolerance=0.005,
        swing_tolerance=0.012,
        sit_back_distance=0.02,
        side_step_speed=0.02,
        jump=JumpSettings(crouch_depth=0.012, takeoff_extension=0.002, height=0.015),
        gaits={
            Gait.WALK: GaitSettings(
                duty_factor=0.6,
                step_height=0.015,
                default_speed=0.08,
                stance_knee_flexion=0.15,
            ),
            Gait.JOG: GaitSettings(
                duty_factor=0.45,
                step_height=0.018,
                default_speed=0.15,
                stance_knee_flexion=0.3,
            ),
            Gait.RUN: GaitSettings(
                duty_factor=0.38,
                step_height=0.02,
                default_speed=0.25,
                stance_knee_flexion=0.35,
            ),
        },
    )


#: Animation settings of every supported hardware variant.
ROBOTS: Final[dict[HardwareVariant, AnimationRobot]] = {
    **{variant: _mini_pi_plus(variant) for variant in MiniPiPlusVariant},
    MicroduckVariant.LEGGED: _microduck(MicroduckVariant.LEGGED),
}


def parse_variant(value: str) -> HardwareVariant:
    """Parse a hardware variant name supported by the animation generators.

    Args:
        value: Variant value such as `PiPlus_S_12L8A0G2H1W` or `microduck`.

    Returns:
        The matching hardware variant.

    Raises:
        KeyError: If the variant is unknown or unsupported.
    """
    for variant in ROBOTS:
        if value in {variant.value, variant.name}:
            return variant
    available = ", ".join(variant.value for variant in ROBOTS)
    raise KeyError(f"Unsupported animation robot '{value}'. Available: {available}")


__all__ = [
    "ROBOTS",
    "AnimationRobot",
    "GaitSettings",
    "JumpSettings",
    "SupportPad",
    "parse_variant",
]
