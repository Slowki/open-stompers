"""Command schedules and resolved gait parameters."""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..robots import AnimationRobot
from ..types import Behavior, Gait
from .cadence import scaled_cadence

#: Behavior label of each gait.
GAIT_BEHAVIORS: Final = {
    Gait.STAND: Behavior.STAND,
    Gait.WALK: Behavior.WALK,
    Gait.JOG: Behavior.JOG,
    Gait.RUN: Behavior.RUN,
}


@dataclass(frozen=True)
class CommandSegment:
    """Constant command held for a duration; unset gait settings use robot defaults."""

    #: Gait used during the segment.
    gait: Gait
    #: Segment length in seconds.
    duration: float
    #: Velocity command `[vx, vy, yaw_rate]` in the heading frame.
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Steps per minute.
    cadence: float | None = None
    #: Fraction of a gait cycle each foot spends on the ground.
    duty_factor: float | None = None
    #: Peak swing foot clearance in meters.
    step_height: float | None = None


@dataclass(frozen=True)
class GaitParameters:
    """Fully resolved gait settings of a segment."""

    #: Gait family.
    gait: Gait
    #: Velocity command `[vx, vy, yaw_rate]` in the heading frame.
    command: NDArray[np.float64]
    #: Steps per minute.
    cadence: float
    #: Fraction of a gait cycle each foot spends on the ground.
    duty_factor: float
    #: Peak swing foot clearance in meters.
    step_height: float
    #: Least knee bend of loaded legs in radians; smaller is straighter.
    stance_knee_flexion: float
    #: Forward lean of the base in radians.
    base_pitch: float
    #: Final fraction of each stance spent rolling onto the toe.
    heel_off: float = 0.0
    #: Toe-down foot pitch reached at liftoff, in radians.
    toe_off_pitch: float = 0.0
    #: Swing progress of the highest foot clearance.
    swing_peak: float = 0.5
    #: Elbow bend in radians, the home pose when unset.
    elbow_flexion: float | None = None
    #: Forward offset of the arm swing center in radians.
    arm_forward_bias: float = 0.0
    #: Largest arm swing either way from the swing center, in radians.
    arm_swing_limit: float = 0.35
    #: Lateral distance between the foot centers in meters, the standing width when unset.
    step_width: float | None = None

    @property
    def step_period(self) -> float:
        """Return the time between consecutive touchdowns in seconds."""
        return 60.0 / self.cadence

    @property
    def swing_duration(self) -> float:
        """Return how long each foot stays in the air per step in seconds."""
        return 2 * self.step_period * (1 - self.duty_factor)

    @property
    def moving(self) -> bool:
        """Return whether the gait takes steps."""
        return self.gait is not Gait.STAND


def default_cadence(robot: AnimationRobot, gait: Gait, speed: float, height: float) -> float:
    """Return the configured cadence of a gait, or the human cadence scaled to the robot.

    Args:
        robot: Robot whose gait settings apply.
        gait: Gait to time; standing uses walking.
        speed: Ground speed in meters per second.
        height: Robot standing height in meters.

    Returns:
        Steps per minute.
    """
    settings = robot.gaits[Gait.WALK if gait is Gait.STAND else gait]
    return settings.cadence if settings.cadence is not None else scaled_cadence(gait, speed, height)


def resolve_segment(
    robot: AnimationRobot, segment: CommandSegment, height: float
) -> GaitParameters:
    """Fill unset segment settings from robot defaults.

    Standing segments use walking timing for the steps that bring the robot to rest.

    Args:
        robot: Robot whose defaults apply.
        segment: Segment to resolve.
        height: Robot standing height in meters, which scales the default cadence.

    Returns:
        The resolved parameters.

    Raises:
        ValueError: If the settings are not physically meaningful.
    """
    if segment.gait is Gait.STAND:
        # Final steps blend from the previous gait into walking timing.
        defaults = robot.gaits[Gait.WALK]
        return GaitParameters(
            gait=Gait.STAND,
            command=np.zeros(3),
            cadence=default_cadence(robot, Gait.WALK, defaults.default_speed, height),
            duty_factor=defaults.duty_factor,
            step_height=defaults.step_height,
            stance_knee_flexion=defaults.stance_knee_flexion,
            base_pitch=0.0,
        )
    defaults = robot.gaits[segment.gait]
    parameters = GaitParameters(
        gait=segment.gait,
        command=np.asarray(segment.command, dtype=np.float64),
        cadence=segment.cadence
        if segment.cadence is not None
        else default_cadence(robot, segment.gait, float(np.hypot(*segment.command[:2])), height),
        duty_factor=segment.duty_factor
        if segment.duty_factor is not None
        else defaults.duty_factor,
        step_height=segment.step_height
        if segment.step_height is not None
        else defaults.step_height,
        stance_knee_flexion=defaults.stance_knee_flexion,
        base_pitch=defaults.base_pitch,
        heel_off=defaults.heel_off,
        toe_off_pitch=defaults.toe_off_pitch,
        swing_peak=defaults.swing_peak,
        step_width=defaults.step_width,
        elbow_flexion=defaults.elbow_flexion,
        arm_forward_bias=defaults.arm_forward_bias,
        arm_swing_limit=defaults.arm_swing_limit,
    )
    if parameters.cadence <= 0 or not 0.2 < parameters.duty_factor < 0.95:
        raise ValueError("Cadence must be positive and the duty factor within (0.2, 0.95)")
    if parameters.step_height < 0 or segment.duration <= 0:
        raise ValueError("Step height and segment duration must be positive")
    return parameters


def blended_parameters(
    parameters: list[GaitParameters],
    boundaries: NDArray[np.float64],
    time: float,
    ramp_steps: float,
) -> GaitParameters:
    """Blend gait timing linearly from the previous segment after each boundary.

    Args:
        parameters: Resolved segment parameters.
        boundaries: Segment start times plus the end time.
        time: Evaluation time in seconds.
        ramp_steps: Ramp length in step periods of the entered segment.

    Returns:
        The segment parameters with cadence, duty factor, and step height blended.
    """
    index = int(
        np.clip(np.searchsorted(boundaries, time, side="right") - 1, 0, len(parameters) - 1)
    )
    current = parameters[index]
    if index == 0 or not parameters[index - 1].moving:
        return current
    previous = parameters[index - 1]
    progress = float(np.clip((time - boundaries[index]) / (ramp_steps * current.step_period), 0, 1))
    # Blend the step period rather than the cadence, keeping the timing smooth.
    period = previous.step_period + progress * (current.step_period - previous.step_period)
    return GaitParameters(
        gait=current.gait,
        command=current.command,
        cadence=60.0 / period,
        duty_factor=previous.duty_factor + progress * (current.duty_factor - previous.duty_factor),
        step_height=previous.step_height + progress * (current.step_height - previous.step_height),
        stance_knee_flexion=current.stance_knee_flexion,
        base_pitch=current.base_pitch,
        heel_off=previous.heel_off + progress * (current.heel_off - previous.heel_off),
        toe_off_pitch=previous.toe_off_pitch
        + progress * (current.toe_off_pitch - previous.toe_off_pitch),
        swing_peak=previous.swing_peak + progress * (current.swing_peak - previous.swing_peak),
        elbow_flexion=current.elbow_flexion,
        arm_forward_bias=previous.arm_forward_bias
        + progress * (current.arm_forward_bias - previous.arm_forward_bias),
        arm_swing_limit=previous.arm_swing_limit
        + progress * (current.arm_swing_limit - previous.arm_swing_limit),
        step_width=(
            previous.step_width + progress * (current.step_width - previous.step_width)
            if previous.step_width is not None and current.step_width is not None
            else current.step_width
        ),
    )


def resolve_schedule(
    robot: AnimationRobot,
    segments: list[CommandSegment] | tuple[CommandSegment, ...],
    height: float,
) -> tuple[list[GaitParameters], NDArray[np.float64]]:
    """Resolve every segment and compute the segment start times.

    Args:
        robot: Robot whose defaults apply.
        segments: Command segments in playback order.
        height: Robot standing height in meters, which scales default cadences.

    Returns:
        Resolved parameters and start times with one extra end time.
    """
    if not segments:
        raise ValueError("A schedule needs at least one segment")
    resolved = [resolve_segment(robot, segment, height) for segment in segments]
    boundaries = np.concatenate(([0.0], np.cumsum([segment.duration for segment in segments])))
    return resolved, boundaries


__all__ = [
    "GAIT_BEHAVIORS",
    "CommandSegment",
    "GaitParameters",
    "blended_parameters",
    "default_cadence",
    "resolve_schedule",
    "resolve_segment",
]
