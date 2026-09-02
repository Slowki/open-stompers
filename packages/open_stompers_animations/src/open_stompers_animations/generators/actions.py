"""One-shot actions: gait transitions, sitting, standing up, and jumping."""

from dataclasses import replace
from typing import Final

import mujoco
import numpy as np
from numpy.typing import NDArray

from open_stompers_hardware_models import HardwareVariant

from ..kinematics import InverseKinematics, standing_target
from ..model import RobotModel, build_robot_model
from ..planning.center_of_mass import GRAVITY, CenterOfMassPlan
from ..planning.commands import CommandSegment, default_cadence, resolve_schedule
from ..planning.footsteps import FootStates
from ..planning.swing import clearance, minimum_jerk
from ..types import DEFAULT_SAMPLE_INTERVAL, Animation, AnimationKind, Behavior, Gait
from .locomotion import (
    PLANNER_SUBDIVISIONS,
    MotionPlan,
    ReachError,
    build_animation,
    frame_indices,
    plan_origin,
    plan_sequence,
)

#: Gait cycles walked or run before the kept part of a transition.
WARM_UP_CYCLES: Final = 3
#: Replanning attempts that lower the center of mass where reach falls short.
_REACH_ATTEMPTS: Final = 8
#: Seconds of standing before and after one-shot motions.
STANDING_HOLD: Final = 0.4
#: Seated center of mass position behind the feet as a fraction of the pad distance.
_SEATED_CENTER_OF_MASS: Final = 0.7


def _cycle(variant: HardwareVariant, gait: Gait) -> float:
    """Return the default two-step cycle duration of a gait in seconds."""
    robot_model = build_robot_model(variant)
    speed = robot_model.robot.gaits[gait].default_speed
    return 120.0 / default_cadence(robot_model.robot, gait, speed, robot_model.height)


def _speed(variant: HardwareVariant, gait: Gait, velocity: float | None) -> float:
    """Return a forward speed, the gait default when omitted."""
    return (
        build_robot_model(variant).robot.gaits[gait].default_speed if velocity is None else velocity
    )


def sequence_action(
    variant: HardwareVariant,
    name: str,
    segments: list[CommandSegment],
    *,
    keep_from: int = 0,
    keep_until: int | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    parameters: dict[str, object] | None = None,
) -> Animation:
    """Generate an action from a command schedule, keeping a window of segments.

    Segments before `keep_from` and from `keep_until` on only bring the robot
    into or out of motion; they are planned but cropped. The kept window is
    moved so it starts at the origin facing forward.

    Args:
        variant: Hardware variant to animate.
        name: Animation name.
        segments: Command schedule starting and ending at rest.
        keep_from: First kept segment.
        keep_until: First cropped trailing segment, or None to keep the rest.
        sample_interval: Time between frames in seconds.
        parameters: Generator settings to record.

    Returns:
        The action animation.
    """
    robot_model = build_robot_model(variant)
    _, boundaries = resolve_schedule(robot_model.robot, segments, robot_model.height)
    stop = None if keep_until is None else float(boundaries[keep_until])
    lowered: tuple[float, ...] = ()
    # Crouch a little wherever a long transition step is out of reach, then replan.
    for _ in range(_REACH_ATTEMPTS):
        plan = plan_sequence(robot_model, segments, sample_interval, lowered)
        indices = frame_indices(plan, sample_interval, float(boundaries[keep_from]), stop)
        try:
            return _sequence_animation(
                robot_model, plan, name, segments, indices, keep_from, keep_until, parameters
            )
        except ReachError as error:
            failure = error
            lowered = (*lowered, error.time)
    raise failure


def _sequence_animation(
    robot_model: RobotModel,
    plan: MotionPlan,
    name: str,
    segments: list[CommandSegment],
    indices: NDArray[np.int64],
    keep_from: int,
    keep_until: int | None,
    parameters: dict[str, object] | None,
) -> Animation:
    """Solve and package the kept frames of a sequence plan."""
    return build_animation(
        robot_model,
        plan,
        name=name,
        kind=AnimationKind.ACTION,
        sample_interval=float(plan.times[PLANNER_SUBDIVISIONS]),
        indices=indices,
        parameters={
            "segments": [
                {
                    "gait": str(segment.gait),
                    "duration": segment.duration,
                    "command": list(segment.command),
                    "cadence": segment.cadence,
                }
                for segment in segments
            ],
            "kept_segments": [keep_from, keep_until],
            **(parameters or {}),
        },
        origin=plan_origin(plan, int(indices[0])),
    )


def start_walking(
    variant: HardwareVariant,
    velocity: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate standing still, then accelerating into a steady walk."""
    speed, cycle = _speed(variant, Gait.WALK, velocity), _cycle(variant, Gait.WALK)
    segments = [
        CommandSegment(Gait.STAND, 1.0),
        CommandSegment(Gait.WALK, WARM_UP_CYCLES * cycle, (speed, 0.0, 0.0)),
        CommandSegment(Gait.WALK, 2 * cycle, (speed, 0.0, 0.0)),
        CommandSegment(Gait.STAND, 2.0),
    ]
    return sequence_action(
        variant, "start_walking", segments, keep_until=2, sample_interval=sample_interval
    )


def stop_walking(
    variant: HardwareVariant,
    velocity: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a steady walk coming to a stop."""
    speed, cycle = _speed(variant, Gait.WALK, velocity), _cycle(variant, Gait.WALK)
    segments = [
        CommandSegment(Gait.STAND, 0.5),
        CommandSegment(Gait.WALK, WARM_UP_CYCLES * cycle, (speed, 0.0, 0.0)),
        CommandSegment(Gait.WALK, cycle, (speed, 0.0, 0.0)),
        CommandSegment(Gait.STAND, 1.0),
    ]
    return sequence_action(
        variant, "stop_walking", segments, keep_from=2, sample_interval=sample_interval
    )


def walk_to_run(
    variant: HardwareVariant,
    walk_velocity: float | None = None,
    run_velocity: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a steady walk accelerating into a run."""
    walking = _speed(variant, Gait.WALK, walk_velocity)
    running = _speed(variant, Gait.RUN, run_velocity)
    walk_cycle, run_cycle = _cycle(variant, Gait.WALK), _cycle(variant, Gait.RUN)
    segments = [
        CommandSegment(Gait.STAND, 0.5),
        CommandSegment(Gait.WALK, WARM_UP_CYCLES * walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.WALK, walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.RUN, 4 * run_cycle, (running, 0.0, 0.0)),
        CommandSegment(Gait.RUN, 2 * run_cycle, (running, 0.0, 0.0)),
        CommandSegment(Gait.STAND, 1.0),
    ]
    return sequence_action(
        variant, "walk_to_run", segments, keep_from=2, keep_until=4, sample_interval=sample_interval
    )


def run_to_walk(
    variant: HardwareVariant,
    run_velocity: float | None = None,
    walk_velocity: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a steady run slowing into a walk."""
    walking = _speed(variant, Gait.WALK, walk_velocity)
    running = _speed(variant, Gait.RUN, run_velocity)
    walk_cycle, run_cycle = _cycle(variant, Gait.WALK), _cycle(variant, Gait.RUN)
    segments = [
        CommandSegment(Gait.STAND, 0.5),
        CommandSegment(Gait.WALK, 2 * walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.RUN, WARM_UP_CYCLES * run_cycle, (running, 0.0, 0.0)),
        CommandSegment(Gait.RUN, run_cycle, (running, 0.0, 0.0)),
        CommandSegment(Gait.WALK, 3 * walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.WALK, walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.STAND, 1.0),
    ]
    return sequence_action(
        variant, "run_to_walk", segments, keep_from=3, keep_until=5, sample_interval=sample_interval
    )


def run_to_stop(
    variant: HardwareVariant,
    velocity: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a steady run braking through a few walking steps to a standstill."""
    running, walking = _speed(variant, Gait.RUN, velocity), _speed(variant, Gait.WALK, None)
    walk_cycle, run_cycle = _cycle(variant, Gait.WALK), _cycle(variant, Gait.RUN)
    segments = [
        CommandSegment(Gait.STAND, 0.5),
        CommandSegment(Gait.WALK, 2 * walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.RUN, WARM_UP_CYCLES * run_cycle, (running, 0.0, 0.0)),
        CommandSegment(Gait.RUN, run_cycle, (running, 0.0, 0.0)),
        # Braking straight from a run overloads the ankles of the trailing leg.
        CommandSegment(Gait.WALK, walk_cycle, (walking, 0.0, 0.0)),
        CommandSegment(Gait.STAND, 1.0),
    ]
    return sequence_action(
        variant, "run_to_stop", segments, keep_from=3, sample_interval=sample_interval
    )


def _standing(robot_model: RobotModel) -> tuple[NDArray[np.float64], mujoco.MjData]:  # ty: ignore[unresolved-attribute]
    """Solve the standing configuration and return it with evaluated kinematics."""
    configuration = (
        InverseKinematics(robot_model)
        .solve(standing_target(robot_model), robot_model.home_configuration)
        .configuration
    )
    data = mujoco.MjData(robot_model.model)  # ty: ignore[unresolved-attribute]
    data.qpos[:] = configuration
    mujoco.mj_kinematics(robot_model.model, data)  # ty: ignore[unresolved-attribute]
    mujoco.mj_comPos(robot_model.model, data)  # ty: ignore[unresolved-attribute]
    return configuration, data


def _static_feet(robot_model: RobotModel, count: int) -> FootStates:
    """Return both soles resting at the standing placements."""
    half_width = robot_model.stance_width / 2
    positions = np.zeros((count, 2, 3))
    positions[:, 0, 1], positions[:, 1, 1] = half_width, -half_width
    return FootStates(
        positions=positions,
        yaws=np.zeros((count, 2)),
        contacts=np.ones((count, 2), dtype=bool),
        swing_fractions=np.zeros((count, 2)),
        pitches=np.zeros((count, 2)),
    )


def _planner_times(duration: float, sample_interval: float) -> NDArray[np.float64]:
    """Return a planner grid covering a duration with whole frames."""
    frames = round(duration / sample_interval) + 1
    return np.arange((frames - 1) * PLANNER_SUBDIVISIONS + 1, dtype=np.float64) * (
        sample_interval / PLANNER_SUBDIVISIONS
    )


def _center_of_mass(
    positions: NDArray[np.float64], interval: float, zero_moment_point: NDArray[np.float64]
) -> CenterOfMassPlan:
    """Wrap sampled center of mass positions into a plan with finite-difference rates."""
    velocities = np.gradient(positions, interval, axis=0)
    return CenterOfMassPlan(
        positions=positions,
        velocities=velocities,
        accelerations=np.gradient(velocities, interval, axis=0),
        zero_moment_point=zero_moment_point,
    )


def _sit_plan(robot_model: RobotModel, duration: float, sample_interval: float) -> MotionPlan:
    """Plan lowering onto the support pad behind the feet."""
    robot = robot_model.robot
    times = _planner_times(2 * STANDING_HOLD + duration, sample_interval)
    count = len(times)
    progress = np.clip((times - STANDING_HOLD) / duration, 0, 1)
    blend = minimum_jerk(progress)[:, None]
    _, data = _standing(robot_model)
    start = data.site_xpos[robot_model.pad_site].copy()
    end = np.array([-robot.sit_back_distance, 0.0, 0.0])
    # Keep the center of mass over the feet until the pad is about to touch down.
    shift = minimum_jerk((progress - 0.75) / 0.25)
    center_of_mass = np.zeros((count, 3))
    center_of_mass[:, 0] = -_SEATED_CENTER_OF_MASS * robot.sit_back_distance * shift
    center_of_mass[:, 2] = data.subtree_com[robot_model.root_body, 2]
    seated = progress >= 1.0
    behaviors = tuple(
        Behavior.SIT if done else Behavior.SIT_DOWN if value > 0 else Behavior.STAND
        for done, value in zip(seated, progress, strict=True)
    )
    return MotionPlan(
        times=times,
        center_of_mass=_center_of_mass(center_of_mass, times[1], center_of_mass[:, :2]),
        feet=_static_feet(robot_model, count),
        base_yaws=np.zeros(count),
        base_pitches=np.zeros(count),
        commands=np.zeros((count, 3)),
        behaviors=behaviors,
        phases=np.zeros(count),
        pad_contacts=seated,
        pad_positions=start + blend * (end - start),
        arm_swing=False,
        # Lean toward keeping the center of mass over the feet without forcing it.
        center_of_mass_axes=(0.05, 1.0, 0.0),
    )


def sit_down(
    variant: HardwareVariant,
    duration: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate lowering from standing onto the support pad.

    Args:
        variant: Hardware variant to animate.
        duration: Seconds of motion between the standing and seated holds.
        sample_interval: Time between frames in seconds.

    Returns:
        The action animation.
    """
    robot_model = build_robot_model(variant)
    motion = 2.0 if duration is None else duration
    plan = _sit_plan(robot_model, motion, sample_interval)
    return build_animation(
        robot_model,
        plan,
        name="sit_down",
        kind=AnimationKind.ACTION,
        sample_interval=sample_interval,
        indices=frame_indices(plan, sample_interval),
        parameters={"generator": "sit_down", "duration": motion},
    )


def stand_up(
    variant: HardwareVariant,
    duration: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate rising from the support pad, the reverse of `sit_down`."""
    seated = sit_down(variant, duration, sample_interval=sample_interval)
    labels = {Behavior.SIT_DOWN: Behavior.STAND_UP}
    return replace(
        seated,
        name="stand_up",
        root_positions=seated.root_positions[::-1].copy(),
        root_quaternions=seated.root_quaternions[::-1].copy(),
        joint_positions=seated.joint_positions[::-1].copy(),
        commands=seated.commands[::-1].copy(),
        behaviors=tuple(labels.get(label, label) for label in reversed(seated.behaviors)),
        phases=seated.phases[::-1].copy(),
        contacts=seated.contacts[::-1].copy(),
        center_of_mass=seated.center_of_mass[::-1].copy(),
        zero_moment_point=seated.zero_moment_point[::-1].copy(),
        parameters={**seated.parameters, "generator": "stand_up"},
    )


def _quintic(
    elapsed: NDArray[np.float64],
    duration: float,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Evaluate the quintic matching position, velocity, and acceleration at both ends."""
    p0, v0, a0 = start
    p1, v1, a1 = end
    time = duration
    matrix = np.array(
        [
            [time**3, time**4, time**5],
            [3 * time**2, 4 * time**3, 5 * time**4],
            [6 * time, 12 * time**2, 20 * time**3],
        ]
    )
    remainder = np.array(
        [
            p1 - (p0 + v0 * time + a0 * time**2 / 2),
            v1 - (v0 + a0 * time),
            a1 - a0,
        ]
    )
    c3, c4, c5 = np.linalg.solve(matrix, remainder)
    t = np.clip(elapsed, 0.0, duration)
    return p0 + v0 * t + a0 * t**2 / 2 + c3 * t**3 + c4 * t**4 + c5 * t**5


def _jump_plan(robot_model: RobotModel, height: float, sample_interval: float) -> MotionPlan:
    """Plan a crouch, push-off, ballistic flight, landing, and recovery in place."""
    settings = robot_model.robot.jump
    standing = robot_model.center_of_mass_height
    crouched = standing - settings.crouch_depth
    takeoff = standing + settings.takeoff_extension
    launch = np.sqrt(2 * GRAVITY * height)
    crouch, push, flight = 0.5, 2.5 * (takeoff - crouched) / launch, 2 * launch / GRAVITY
    absorb, recover = 2.5 * (takeoff - crouched) / launch, 0.6
    takeoff_time = STANDING_HOLD + crouch + push
    landing_time = takeoff_time + flight
    duration = landing_time + absorb + recover + STANDING_HOLD
    times = _planner_times(duration, sample_interval)
    heights = np.full(len(times), standing)
    phases = (
        (STANDING_HOLD, crouch, (standing, 0.0, 0.0), (crouched, 0.0, 0.0)),
        (STANDING_HOLD + crouch, push, (crouched, 0.0, 0.0), (takeoff, launch, -GRAVITY)),
        (landing_time, absorb, (takeoff, -launch, -GRAVITY), (crouched, 0.0, 0.0)),
        (landing_time + absorb, recover, (crouched, 0.0, 0.0), (standing, 0.0, 0.0)),
    )
    for begin, length, first, last in phases:
        window = (times >= begin) & (times <= begin + length)
        heights[window] = _quintic(times[window] - begin, length, first, last)
    airborne = (times > takeoff_time) & (times < landing_time)
    rise = times[airborne] - takeoff_time
    heights[airborne] = takeoff + launch * rise - GRAVITY * rise**2 / 2
    velocities = np.gradient(heights, times)
    accelerations = np.gradient(velocities, times)
    feet = _static_feet(robot_model, len(times))
    flight_progress = np.clip((times - takeoff_time) / flight, 0, 1)
    # Feet stay under the hips and tuck slightly at the apex.
    lift = (heights - takeoff) + 0.3 * height * clearance(flight_progress)
    feet.positions[airborne, :, 2] = np.maximum(lift[airborne], 0.0)[:, None]
    feet.contacts[airborne] = False
    center_of_mass = np.zeros((len(times), 3))
    center_of_mass[:, 2] = heights
    zeros = np.zeros(len(times))
    moving = (times >= STANDING_HOLD) & (times <= duration - STANDING_HOLD)
    return MotionPlan(
        times=times,
        center_of_mass=CenterOfMassPlan(
            positions=center_of_mass,
            velocities=np.stack((zeros, zeros, velocities), axis=1),
            accelerations=np.stack((zeros, zeros, accelerations), axis=1),
            zero_moment_point=np.zeros((len(times), 2)),
        ),
        feet=feet,
        base_yaws=zeros.copy(),
        base_pitches=zeros.copy(),
        commands=np.zeros((len(times), 3)),
        behaviors=tuple(Behavior.JUMP if value else Behavior.STAND for value in moving),
        phases=zeros.copy(),
        pad_contacts=np.zeros(len(times), dtype=bool),
        arm_swing=False,
    )


def jump_vertical(
    variant: HardwareVariant,
    height: float | None = None,
    *,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> Animation:
    """Generate a vertical jump in place.

    Args:
        variant: Hardware variant to animate.
        height: Rise of the center of mass above takeoff in meters.
        sample_interval: Time between frames in seconds.

    Returns:
        The action animation.
    """
    robot_model = build_robot_model(variant)
    apex = robot_model.robot.jump.height if height is None else height
    plan = _jump_plan(robot_model, apex, sample_interval)
    return build_animation(
        robot_model,
        plan,
        name="jump_vertical",
        kind=AnimationKind.ACTION,
        sample_interval=sample_interval,
        indices=frame_indices(plan, sample_interval),
        parameters={"generator": "jump_vertical", "height": apex},
    )


__all__ = [
    "jump_vertical",
    "run_to_stop",
    "run_to_walk",
    "sequence_action",
    "sit_down",
    "stand_up",
    "start_walking",
    "stop_walking",
    "walk_to_run",
]
