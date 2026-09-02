"""Heading paths, footstep placement, foot trajectories, and support references."""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Final, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..robots import AnimationRobot
from .commands import GaitParameters, blended_parameters
from .swing import minimum_jerk, swing_pose, wrap_angle

#: Sampling interval of integrated heading paths in seconds.
PATH_INTERVAL: Final = 0.002
#: Step periods over which the planning twist ramps toward a new command.
RAMP_STEPS: Final = 3
#: Fraction of the reach limits that transition steps aim to stay within.
_REACH_MARGIN: Final = 0.98


class Side(IntEnum):
    """Foot side, also the foot index in planning arrays."""

    LEFT = 0
    RIGHT = 1


def integrate_twist(
    pose: NDArray[np.float64], twist: NDArray[np.float64], duration: ArrayLike
) -> NDArray[np.float64]:
    """Move planar poses along a constant heading-frame twist.

    Args:
        pose: Start pose `[x, y, yaw]`.
        twist: Heading-frame `[vx, vy, yaw_rate]`.
        duration: Elapsed times in seconds shaped (samples,).

    Returns:
        Poses shaped (samples, 3) with continuous yaw.
    """
    elapsed = np.atleast_1d(np.asarray(duration, dtype=np.float64))
    forward, lateral, rate = twist
    turned = rate * elapsed
    if abs(rate) < 1e-9:
        local_x, local_y = forward * elapsed, lateral * elapsed
    else:
        local_x = (forward * np.sin(turned) + lateral * (np.cos(turned) - 1)) / rate
        local_y = (forward * (1 - np.cos(turned)) + lateral * np.sin(turned)) / rate
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    return np.stack(
        (
            pose[0] + cosine * local_x - sine * local_y,
            pose[1] + sine * local_x + cosine * local_y,
            pose[2] + turned,
        ),
        axis=1,
    )


class HeadingPath(Protocol):
    """Planar path followed by the center of mass and heading."""

    def pose(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return `[x, y, yaw]` poses shaped (samples, 3) with continuous yaw."""
        ...


@dataclass(frozen=True)
class ConstantTwistPath:
    """Path of a constant heading-frame twist starting at the origin."""

    #: Heading-frame `[vx, vy, yaw_rate]`.
    twist: NDArray[np.float64]

    def pose(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return poses along the path at arbitrary, possibly negative, times."""
        return integrate_twist(np.zeros(3), self.twist, times)


@dataclass(frozen=True)
class SampledPath:
    """Path integrated from a time-varying twist, holding still outside its span."""

    #: Sample times in seconds shaped (samples,).
    times: NDArray[np.float64]
    #: Poses shaped (samples, 3) with continuous yaw.
    poses: NDArray[np.float64]

    def pose(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return linearly interpolated poses."""
        requested = np.atleast_1d(np.asarray(times, dtype=np.float64))
        return np.stack(
            [np.interp(requested, self.times, self.poses[:, axis]) for axis in range(3)], axis=1
        )


def schedule_twist(
    parameters: list[GaitParameters], boundaries: NDArray[np.float64], times: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the planning twist, ramping over `RAMP_STEPS` step periods after each boundary.

    Args:
        parameters: Resolved segment parameters.
        boundaries: Segment start times plus the end time.
        times: Evaluation times in seconds.

    Returns:
        Heading-frame twists shaped (samples, 3).
    """
    requested = np.atleast_1d(np.asarray(times, dtype=np.float64))
    twist = np.tile(parameters[0].command, (len(requested), 1))
    for index in range(1, len(parameters)):
        start = boundaries[index]
        ramp = RAMP_STEPS * parameters[index].step_period
        progress = np.clip((requested - start) / ramp, 0, 1)[:, None]
        twist = twist + progress * (parameters[index].command - parameters[index - 1].command)
    return twist


def integrate_schedule(
    parameters: list[GaitParameters], boundaries: NDArray[np.float64], end_time: float
) -> SampledPath:
    """Integrate the planning twist of a schedule into a heading path.

    Args:
        parameters: Resolved segment parameters.
        boundaries: Segment start times plus the end time.
        end_time: Last time the path must cover, in seconds.

    Returns:
        The sampled path starting at the origin.
    """
    times = np.arange(0.0, end_time + 2 * PATH_INTERVAL, PATH_INTERVAL)
    midpoints = schedule_twist(parameters, boundaries, times[:-1] + PATH_INTERVAL / 2)
    poses = np.zeros((len(times), 3))
    for index in range(len(midpoints)):
        poses[index + 1] = integrate_twist(poses[index], midpoints[index], PATH_INTERVAL)[0]
    return SampledPath(times, poses)


@dataclass(frozen=True)
class Step:
    """Single foot movement between two ground placements."""

    #: Moving foot.
    side: Side
    #: Time the foot leaves the ground in seconds.
    lift_time: float
    #: Time the foot lands in seconds.
    touchdown_time: float
    #: Liftoff placement `[x, y, yaw]`.
    start: NDArray[np.float64]
    #: Touchdown placement `[x, y, yaw]`.
    end: NDArray[np.float64]
    #: Peak clearance in meters.
    height: float
    #: Fraction of a gait cycle the foot spends on the ground.
    duty_factor: float
    #: Seconds before liftoff spent rolling onto the toe.
    heel_off_duration: float = 0.0
    #: Toe-down foot pitch reached at liftoff, in radians.
    toe_off_pitch: float = 0.0
    #: Swing progress of the highest clearance.
    swing_peak: float = 0.5


@dataclass(frozen=True)
class FootStates:
    """Sole trajectories of both feet."""

    #: Sole centers shaped (samples, 2, 3).
    positions: NDArray[np.float64]
    #: Sole headings shaped (samples, 2).
    yaws: NDArray[np.float64]
    #: Ground contact flags shaped (samples, 2).
    contacts: NDArray[np.bool_]
    #: Swing progress in [0, 1] shaped (samples, 2), zero while in contact.
    swing_fractions: NDArray[np.float64]
    #: Toe-down sole pitches in radians shaped (samples, 2).
    pitches: NDArray[np.float64]


@dataclass(frozen=True)
class FootstepPlan:
    """Ordered steps of both feet starting from initial placements."""

    #: Placements before the first step shaped (2, 3).
    initial: NDArray[np.float64]
    #: Steps ordered by touchdown time.
    steps: tuple[Step, ...]

    def evaluate(self, times: ArrayLike, sole_half_lengths: ArrayLike = (0.0, 0.0)) -> FootStates:
        """Evaluate both sole trajectories.

        Feet with a toe-off pitch roll about their toe edge before liftoff and
        level out again during the first half of the swing.

        Args:
            times: Evaluation times in seconds shaped (samples,).
            sole_half_lengths: Distance from each sole center to its toe edge in meters.

        Returns:
            Sole positions, headings, pitches, and contact flags.
        """
        requested = np.atleast_1d(np.asarray(times, dtype=np.float64))
        reaches = np.asarray(sole_half_lengths, dtype=np.float64)
        positions = np.zeros((len(requested), 2, 3))
        yaws = np.zeros((len(requested), 2))
        contacts = np.ones((len(requested), 2), dtype=bool)
        fractions = np.zeros((len(requested), 2))
        pitches = np.zeros((len(requested), 2))
        for side in Side:
            positions[:, side, :2] = self.initial[side, :2]
            yaws[:, side] = self.initial[side, 2]
            for step in (step for step in self.steps if step.side is side):
                if step.heel_off_duration > 0:
                    start = step.lift_time - step.heel_off_duration
                    rolling = (requested > start) & (requested <= step.lift_time)
                    pitches[rolling, side] = step.toe_off_pitch * minimum_jerk(
                        (requested[rolling] - start) / step.heel_off_duration
                    )
                landed = requested >= step.touchdown_time
                positions[landed, side, :2] = step.end[:2]
                positions[landed, side, 2] = 0.0
                yaws[landed, side] = step.end[2]
                swinging = (requested > step.lift_time) & ~landed
                if swinging.any():
                    fraction = (requested[swinging] - step.lift_time) / (
                        step.touchdown_time - step.lift_time
                    )
                    swing_positions, swing_yaws = swing_pose(
                        step.start, step.end, step.height, fraction, step.swing_peak
                    )
                    positions[swinging, side] = swing_positions
                    yaws[swinging, side] = swing_yaws
                    contacts[swinging, side] = False
                    fractions[swinging, side] = fraction
                    if step.heel_off_duration > 0:
                        pitches[swinging, side] = step.toe_off_pitch * (
                            1 - minimum_jerk(2 * fraction)
                        )
        # Pivoting about the toe edge raises the sole center and moves it forward.
        for side in Side:
            forward = reaches[side] * (1 - np.cos(pitches[:, side]))
            positions[:, side, 0] += np.cos(yaws[:, side]) * forward
            positions[:, side, 1] += np.sin(yaws[:, side]) * forward
            positions[:, side, 2] += reaches[side] * np.sin(pitches[:, side])
        return FootStates(positions, yaws, contacts, fractions, pitches)

    def phase(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return the gait phase since the latest left touchdown, zero when not stepping."""
        requested = np.atleast_1d(np.asarray(times, dtype=np.float64))
        touchdowns = np.array(
            [step.touchdown_time for step in self.steps if step.side is Side.LEFT]
        )
        phase = np.zeros(len(requested))
        if len(touchdowns) < 2:
            return phase
        following = np.searchsorted(touchdowns, requested, side="right")
        inside = (following > 0) & (following < len(touchdowns))
        previous = touchdowns[np.clip(following - 1, 0, len(touchdowns) - 1)]
        upcoming = touchdowns[np.clip(following, 0, len(touchdowns) - 1)]
        phase[inside] = (requested[inside] - previous[inside]) / (
            upcoming[inside] - previous[inside]
        )
        return np.mod(phase, 1.0)


def _half_width(gait: GaitParameters, standing: float, minimum: float) -> float:
    """Return half the lateral foot separation, widened if sideways steps would collide."""
    nominal = standing if gait.step_width is None else gait.step_width / 2
    # Sideways steps alternate between the nominal width plus and minus this shift.
    shift = abs(float(gait.command[1])) * gait.step_period / 2
    return max(nominal, minimum / 2 + shift)


def _placement(path: HeadingPath, side: Side, time: float, half_width: float) -> NDArray:
    """Place a foot beside the path pose at a time."""
    x, y, yaw = path.pose(time)[0]
    offset = half_width if side is Side.LEFT else -half_width
    return np.array([x - np.sin(yaw) * offset, y + np.cos(yaw) * offset, yaw])


def _heel_off_duration(gait: GaitParameters) -> float:
    """Return how long a foot rolls onto its toe before liftoff."""
    return gait.heel_off * 2 * gait.step_period * gait.duty_factor


def _stance_sweep(path: HeadingPath, touchdown: float, gait: GaitParameters) -> float:
    """Return how far the body moves over a foot planted for one stance."""
    stance = path.pose([touchdown, touchdown + 2 * gait.step_period * gait.duty_factor])
    return float(np.linalg.norm(stance[1, :2] - stance[0, :2]))


def _validate_step(
    robot: AnimationRobot, step: Step, other: NDArray[np.float64], sweep: float
) -> None:
    """Reject steps beyond the robot reach or crossing the other foot."""
    travel = float(np.linalg.norm(step.end[:2] - step.start[:2]))
    if travel > robot.maximum_foot_travel + 1e-9:
        raise ValueError(
            f"A {travel:.3f} m step exceeds the {robot.maximum_foot_travel:.3f} m foot travel "
            "limit; lower the speed or raise the cadence"
        )
    if sweep > robot.maximum_stance_sweep + 1e-9:
        raise ValueError(
            f"A {sweep:.3f} m stance sweep exceeds the {robot.maximum_stance_sweep:.3f} m limit; "
            "lower the speed or raise the cadence"
        )
    cosine, sine = np.cos(step.end[2]), np.sin(step.end[2])
    offset = other[:2] - step.end[:2]
    lateral = -sine * offset[0] + cosine * offset[1]
    separation = lateral if step.side is Side.RIGHT else -lateral
    if separation < robot.minimum_foot_separation - 1e-9:
        raise ValueError("Feet would cross or collide; lower the lateral speed or turn rate")
    if abs(wrap_angle(step.end[2] - other[2])) > robot.maximum_foot_yaw_difference + 1e-9:
        raise ValueError("Foot yaw difference too large; lower the turn rate")


def _within_reach(
    robot: AnimationRobot,
    gait: GaitParameters,
    path: HeadingPath,
    side: Side,
    last_touchdown: float,
    start: NDArray[np.float64],
    half_width: float,
) -> GaitParameters:
    """Shorten the step period while accelerating until the step fits the robot reach."""
    for _ in range(20):
        touchdown = last_touchdown + gait.step_period
        end = _placement(
            path,
            side,
            touchdown + gait.step_period * gait.duty_factor,
            _half_width(gait, half_width, robot.minimum_foot_separation),
        )
        travel = float(np.linalg.norm(end[:2] - start[:2]))
        if (
            travel <= _REACH_MARGIN * robot.maximum_foot_travel
            and _stance_sweep(path, touchdown, gait) <= _REACH_MARGIN * robot.maximum_stance_sweep
        ):
            break
        gait = replace(gait, cadence=gait.cadence / 0.95)
    return gait


def _leading_side(command: NDArray[np.float64]) -> Side:
    """Choose the first stepping foot so it leads in the direction of travel."""
    if abs(command[1]) > 1e-6:
        return Side.LEFT if command[1] > 0 else Side.RIGHT
    if abs(command[2]) > 1e-6:
        return Side.LEFT if command[2] > 0 else Side.RIGHT
    return Side.LEFT


def plan_loop_footsteps(
    robot: AnimationRobot,
    parameters: GaitParameters,
    half_width: float,
    *,
    cycles: int = 3,
) -> tuple[FootstepPlan, ConstantTwistPath]:
    """Plan periodic steps around the cycle that starts with a left touchdown at zero.

    Args:
        robot: Robot whose reach limits apply.
        parameters: Gait of the loop.
        half_width: Half of the standing lateral foot separation in meters, used by
            gaits without their own step width.
        cycles: Cycles of steps planned on each side of the reference cycle.

    Returns:
        The footstep plan and its heading path.
    """
    path = ConstantTwistPath(parameters.command)
    period = parameters.step_period
    settle = period * parameters.duty_factor
    count = 2 * cycles

    def side_of(index: int) -> Side:
        return Side.LEFT if index % 2 == 0 else Side.RIGHT

    width = _half_width(parameters, half_width, robot.minimum_foot_separation)
    placements = {
        index: _placement(path, side_of(index), index * period + settle, width)
        for index in range(-count - 2, count + 2)
    }
    steps = []
    for index in range(-count, count + 2):
        touchdown = index * period
        step = Step(
            side=side_of(index),
            lift_time=touchdown - parameters.swing_duration,
            touchdown_time=touchdown,
            start=placements[index - 2],
            end=placements[index],
            height=parameters.step_height,
            duty_factor=parameters.duty_factor,
            heel_off_duration=_heel_off_duration(parameters),
            toe_off_pitch=parameters.toe_off_pitch,
            swing_peak=parameters.swing_peak,
        )
        _validate_step(
            robot, step, placements[index - 1], _stance_sweep(path, touchdown, parameters)
        )
        steps.append(step)
    initial = np.stack((placements[-count - 2], placements[-count - 1]))
    return FootstepPlan(initial, tuple(steps)), path


def plan_sequence_footsteps(
    robot: AnimationRobot,
    parameters: list[GaitParameters],
    boundaries: NDArray[np.float64],
    path: HeadingPath,
    half_width: float,
) -> FootstepPlan:
    """Plan steps for a command schedule that starts and ends standing.

    Stepping starts one step period after a moving segment begins, which leaves
    time to shift weight. After a standing segment begins the robot keeps
    stepping until both feet rest beside the stopped path.

    Args:
        robot: Robot whose reach limits apply.
        parameters: Resolved segment parameters.
        boundaries: Segment start times plus the end time.
        path: Heading path of the center of mass.
        half_width: Half of the standing lateral foot separation in meters, used by
            gaits without their own step width.

    Returns:
        The footstep plan.
    """

    def segment_at(time: float) -> int:
        return int(
            np.clip(np.searchsorted(boundaries, time, side="right") - 1, 0, len(parameters) - 1)
        )

    current = np.stack([_placement(path, side, 0.0, half_width) for side in Side])
    initial = current.copy()
    steps: list[Step] = []
    last_touchdown: float | None = None
    side = Side.LEFT
    settled = 0
    cursor = 0.0
    first = False
    while True:
        if last_touchdown is None:
            moving = [
                index
                for index, values in enumerate(parameters)
                if values.moving and boundaries[index + 1] > cursor
            ]
            if not moving:
                break
            index = moving[0]
            gait = parameters[index]
            side = _leading_side(gait.command)
            lift = max(boundaries[index], cursor) + gait.step_period
            touchdown = lift + gait.swing_duration
            first = True
        else:
            index = segment_at(last_touchdown)
            gait = blended_parameters(parameters, boundaries, last_touchdown, RAMP_STEPS)
            # Time the step with the gait blended at its own touchdown.
            for _ in range(4):
                upcoming = last_touchdown + gait.step_period
                gait = blended_parameters(parameters, boundaries, upcoming, RAMP_STEPS)
            if not gait.moving:
                ramp_end = boundaries[index] + RAMP_STEPS * gait.step_period
                settled = (
                    settled + 1
                    if last_touchdown + gait.step_period * gait.duty_factor >= ramp_end
                    else 0
                )
                if settled >= 2:
                    cursor, last_touchdown, settled = last_touchdown, None, 0
                    continue
            gait = _within_reach(robot, gait, path, side, last_touchdown, current[side], half_width)
            touchdown = last_touchdown + gait.step_period
            lift = touchdown - gait.swing_duration
            if first:
                # Never leave the ground with both feet on the way out of standing.
                lift = max(lift, last_touchdown)
            first = False
        placement = _placement(
            path,
            side,
            touchdown + gait.step_period * gait.duty_factor,
            _half_width(gait, half_width, robot.minimum_foot_separation),
        )
        previous_same = next((step for step in reversed(steps) if step.side is side), None)
        if previous_same is not None:
            lift = max(lift, previous_same.touchdown_time + 1e-3)
        step = Step(
            side,
            lift,
            touchdown,
            current[side].copy(),
            placement,
            gait.step_height,
            gait.duty_factor,
            _heel_off_duration(gait),
            gait.toe_off_pitch,
            gait.swing_peak,
        )
        _validate_step(robot, step, current[1 - side], _stance_sweep(path, touchdown, gait))
        steps.append(step)
        current[side] = placement
        last_touchdown = touchdown
        side = Side(1 - side)
    return FootstepPlan(initial, tuple(steps))


def support_reference(
    times: NDArray[np.float64],
    states: FootStates,
    shift_duration: float,
    inset: float = 0.0,
) -> NDArray[np.float64]:
    """Build the zero moment point reference from the contact pattern.

    The reference sits at the stance sole center during single support, moves
    between soles with a minimum jerk profile during double support, and returns
    to the midpoint between the soles while standing still.

    Args:
        times: Uniform evaluation grid in seconds shaped (samples,).
        states: Foot states on the grid.
        shift_duration: Time to move weight onto a foot before the first liftoff
            and back to the midpoint after the last touchdown, in seconds.
        inset: Lateral shift of the single support point toward the other foot,
            in meters, which reduces sideways sway.

    Returns:
        Reference points shaped (samples, 2).
    """
    single = states.contacts.sum(axis=1) == 1
    midpoint = states.positions[:, :, :2].mean(axis=1)
    knot_times: list[float] = []
    knot_points: list[NDArray] = []
    index = 0
    while index < len(times):
        if not single[index]:
            index += 1
            continue
        start = index
        foot = int(np.flatnonzero(states.contacts[index])[0])
        while index < len(times) and single[index] and states.contacts[index, foot]:
            index += 1
        yaw = states.yaws[start, foot]
        inward = (1 if foot == Side.RIGHT else -1) * np.array([-np.sin(yaw), np.cos(yaw)])
        point = states.positions[start, foot, :2] + inset * inward
        knot_times += [times[start], times[index - 1]]
        knot_points += [point, point]
    if not knot_times:
        return midpoint
    if states.contacts[0].all() and knot_times[0] - shift_duration > times[0]:
        knot_times = [times[0], knot_times[0] - shift_duration, *knot_times]
        knot_points = [midpoint[0], midpoint[0], *knot_points]
    if states.contacts[-1].all() and knot_times[-1] + shift_duration < times[-1]:
        knot_times += [knot_times[-1] + shift_duration, times[-1]]
        knot_points += [midpoint[-1], midpoint[-1]]
    return interpolate_knots(np.asarray(knot_times), np.asarray(knot_points), times)


def interpolate_knots(
    knot_times: NDArray[np.float64], knot_values: NDArray[np.float64], times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolate between knots with minimum jerk transitions, holding the ends.

    Args:
        knot_times: Non-decreasing knot times shaped (knots,).
        knot_values: Knot values shaped (knots, dimensions).
        times: Evaluation times shaped (samples,).

    Returns:
        Interpolated values shaped (samples, dimensions).
    """
    following = np.clip(np.searchsorted(knot_times, times, side="right"), 1, len(knot_times) - 1)
    previous = following - 1
    span = np.maximum(knot_times[following] - knot_times[previous], 1e-12)
    fraction = minimum_jerk((times - knot_times[previous]) / span)[:, None]
    values = knot_values[previous] + fraction * (knot_values[following] - knot_values[previous])
    values[times <= knot_times[0]] = knot_values[0]
    values[times >= knot_times[-1]] = knot_values[-1]
    return values


__all__ = [
    "ConstantTwistPath",
    "FootStates",
    "FootstepPlan",
    "HeadingPath",
    "SampledPath",
    "Side",
    "Step",
    "integrate_schedule",
    "integrate_twist",
    "interpolate_knots",
    "plan_loop_footsteps",
    "plan_sequence_footsteps",
    "schedule_twist",
    "support_reference",
]
