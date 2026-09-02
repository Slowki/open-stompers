"""Steady-state locomotion loops."""

import numpy as np

from open_stompers_hardware_models import HardwareVariant

from ..model import build_robot_model
from ..planning.commands import CommandSegment, resolve_segment
from ..types import DEFAULT_SAMPLE_INTERVAL, Animation, AnimationKind, Gait
from .locomotion import PLANNER_SUBDIVISIONS, build_animation, plan_loop


def locomotion_loop(
    variant: HardwareVariant,
    gait: Gait,
    command: tuple[float, float, float],
    *,
    cadence: float | None = None,
    duty_factor: float | None = None,
    step_height: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    name: str | None = None,
) -> Animation:
    """Generate one seamless cycle of steady locomotion.

    The average base velocity over the cycle equals `command`. The cadence is
    rounded so the cycle spans a whole number of frames; the parameters record
    the cadence actually used.

    Args:
        variant: Hardware variant to animate.
        gait: Walking, jogging, or running.
        command: Velocity command `[vx, vy, yaw_rate]` in the heading frame.
        cadence: Steps per minute; when omitted, the average human cadence at the
            dynamically similar speed, scaled by the robot height.
        duty_factor: Fraction of a cycle each foot is on the ground.
        step_height: Peak swing foot clearance in meters.
        sample_interval: Time between frames in seconds.
        name: Animation name, derived from the settings when omitted.

    Returns:
        The loop animation.
    """
    if gait is Gait.STAND:
        raise ValueError("Loops need a moving gait")
    robot_model = build_robot_model(variant)
    segment = CommandSegment(gait, 1.0, command, cadence, duty_factor, step_height)
    requested = resolve_segment(robot_model.robot, segment, robot_model.height)
    plan, offset, parameters = plan_loop(robot_model, requested, sample_interval)
    frames = len(plan.times) // PLANNER_SUBDIVISIONS
    forward, lateral, rate = command
    label = name or (
        f"{gait}_x{forward:+.2f}_y{lateral:+.2f}_yaw{rate:+.2f}_{parameters.cadence:.0f}spm"
    )
    return build_animation(
        robot_model,
        plan,
        name=label,
        kind=AnimationKind.LOOP,
        sample_interval=sample_interval,
        indices=np.arange(frames) * PLANNER_SUBDIVISIONS,
        loop_offset=offset,
        parameters={
            "generator": "locomotion_loop",
            "gait": str(gait),
            "command": [float(value) for value in command],
            "requested_cadence": requested.cadence,
            "cadence": parameters.cadence,
            "duty_factor": parameters.duty_factor,
            "step_height": parameters.step_height,
            "stance_knee_flexion": parameters.stance_knee_flexion,
            "arm_swing_gain": float(plan.arm_swing_gains[0])
            if plan.arm_swing_gains is not None
            else 0.0,
            "step_width": parameters.step_width,
        },
    )


def walk(
    variant: HardwareVariant,
    velocity: float | None = None,
    *,
    lateral_velocity: float = 0.0,
    yaw_rate: float = 0.0,
    cadence: float | None = None,
    step_height: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a walking loop, at the robot default speed when `velocity` is omitted."""
    speed = _default_speed(variant, Gait.WALK) if velocity is None else velocity
    return locomotion_loop(
        variant,
        Gait.WALK,
        (speed, lateral_velocity, yaw_rate),
        cadence=cadence,
        step_height=step_height,
        sample_interval=sample_interval,
    )


def jog(
    variant: HardwareVariant,
    velocity: float | None = None,
    *,
    yaw_rate: float = 0.0,
    cadence: float | None = None,
    step_height: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a jogging loop, at the robot default speed when `velocity` is omitted."""
    speed = _default_speed(variant, Gait.JOG) if velocity is None else velocity
    return locomotion_loop(
        variant,
        Gait.JOG,
        (speed, 0.0, yaw_rate),
        cadence=cadence,
        step_height=step_height,
        sample_interval=sample_interval,
    )


def run(
    variant: HardwareVariant,
    velocity: float | None = None,
    *,
    yaw_rate: float = 0.0,
    cadence: float | None = None,
    step_height: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a running loop, at the robot default speed when `velocity` is omitted."""
    speed = _default_speed(variant, Gait.RUN) if velocity is None else velocity
    return locomotion_loop(
        variant,
        Gait.RUN,
        (speed, 0.0, yaw_rate),
        cadence=cadence,
        step_height=step_height,
        sample_interval=sample_interval,
    )


def turn_in_place(
    variant: HardwareVariant,
    yaw_rate: float = 0.5,
    *,
    cadence: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a loop of stepping on the spot while turning."""
    return locomotion_loop(
        variant, Gait.WALK, (0.0, 0.0, yaw_rate), cadence=cadence, sample_interval=sample_interval
    )


def turn_while_walking(
    variant: HardwareVariant,
    velocity: float | None = None,
    yaw_rate: float = 0.3,
    *,
    cadence: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a loop of walking along a curve."""
    return walk(
        variant,
        velocity,
        yaw_rate=yaw_rate,
        cadence=cadence,
        sample_interval=sample_interval,
    )


def side_step(
    variant: HardwareVariant,
    lateral_velocity: float | None = None,
    *,
    cadence: float | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a loop of stepping sideways; positive velocities move left."""
    if lateral_velocity is None:
        lateral_velocity = build_robot_model(variant).robot.side_step_speed
    return locomotion_loop(
        variant,
        Gait.WALK,
        (0.0, lateral_velocity, 0.0),
        cadence=cadence,
        sample_interval=sample_interval,
    )


def _default_speed(variant: HardwareVariant, gait: Gait) -> float:
    """Return the default forward speed of a gait."""
    return build_robot_model(variant).robot.gaits[gait].default_speed


__all__ = [
    "jog",
    "locomotion_loop",
    "run",
    "side_step",
    "turn_in_place",
    "turn_while_walking",
    "walk",
]
