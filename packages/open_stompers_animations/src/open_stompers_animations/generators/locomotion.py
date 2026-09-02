"""Shared pipeline from contact plans to animations."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from typing import Final

import mujoco
import numpy as np
from numpy.typing import NDArray

from open_stompers_hardware_models import HardwareVariant

from ..kinematics import InverseKinematics, PoseTarget, standing_target, yaw_quaternion
from ..model import RobotModel, build_robot_model
from ..planning.center_of_mass import GRAVITY, CenterOfMassPlan, plan_center_of_mass, plan_vertical
from ..planning.commands import (
    GAIT_BEHAVIORS,
    CommandSegment,
    GaitParameters,
    resolve_schedule,
    resolve_segment,
)
from ..planning.footsteps import (
    RAMP_STEPS,
    FootStates,
    FootstepPlan,
    integrate_schedule,
    plan_loop_footsteps,
    plan_sequence_footsteps,
    schedule_twist,
    support_reference,
)
from ..planning.swing import minimum_jerk
from ..types import (
    DEFAULT_SAMPLE_INTERVAL,
    Animation,
    AnimationKind,
    Behavior,
    Gait,
    LoopOffset,
)

#: Planner samples per animation frame.
PLANNER_SUBDIVISIONS: Final = 4
#: Seconds the robot stands still after its last step.
FINAL_HOLD: Final = 0.5
#: Largest arm swing in radians for plans without per-sample limits.
_DEFAULT_ARM_LIMIT: Final = 0.35
#: Largest arm swing gain as a multiple of the robot default.
_ARM_GAIN_SCALE_LIMIT: Final = 4.0
#: Passes discovering the body height with free inverse kinematics and replanning.
_HEIGHT_PASSES: Final = 2
#: Passes lowering the height target where smoothing overshoots the discovered height.
_LOWERING_ITERATIONS: Final = 10
#: Stance knee target weight while the solver chooses the body height.
_DISCOVERY_KNEE_WEIGHT: Final = 1.0
#: Time over which stance knee targets fade in and out, in seconds.
_KNEE_FADE: Final = 0.08
#: Stance knee target weight while tracking the planned height.
_TRACKING_KNEE_WEIGHT: Final = 0.01
#: Swing progress over which a foot is released after liftoff and caught before touchdown.
_SWING_RELEASE: Final = 0.12
#: Sole position weight kept at mid swing.
_SWING_POSITION_WEIGHT: Final = 0.03
#: Swing position weight scale while both feet are airborne.
_FLIGHT_SWING_SCALE: Final = 0.05
#: Support pad height below which it counts as touching, in meters.
_PAD_CONTACT_HEIGHT: Final = 2e-3
#: Center of mass lowering per reach failure, relative to the standing height.
_REACH_LOWERING: Final = 0.03
#: Standard deviation of the lowering window in seconds.
_REACH_LOWERING_WIDTH: Final = 0.4
#: Sole orientation weight kept at mid swing, letting the foot pitch freely.
_SWING_ORIENTATION_WEIGHT: Final = 0.05


class ReachError(RuntimeError):
    """Inverse kinematics could not reach a planned target."""

    def __init__(self, error: float, time: float) -> None:
        """Describe the missed target.

        Args:
            error: Remaining task error in meters.
            time: Plan time of the failing frame in seconds.
        """
        super().__init__(
            f"Inverse kinematics missed a target by {error * 1000:.1f} mm at {time:.2f} s; "
            "the motion exceeds the robot reach"
        )
        self.error = error
        self.time = time


@dataclass(frozen=True)
class MotionPlan:
    """Task-space plan on the planner grid."""

    #: Planner sample times in seconds shaped (samples,).
    times: NDArray[np.float64]
    #: Center of mass trajectory.
    center_of_mass: CenterOfMassPlan
    #: Sole trajectories and foot contacts.
    feet: FootStates
    #: Base headings shaped (samples,).
    base_yaws: NDArray[np.float64]
    #: Base forward leans shaped (samples,).
    base_pitches: NDArray[np.float64]
    #: Velocity command labels shaped (samples, 3).
    commands: NDArray[np.float64]
    #: Behavior labels.
    behaviors: tuple[Behavior, ...]
    #: Gait phases shaped (samples,).
    phases: NDArray[np.float64]
    #: Support pad contact flags shaped (samples,).
    pad_contacts: NDArray[np.bool_]
    #: Support pad bottom targets shaped (samples, 3), if the pad is driven.
    pad_positions: NDArray[np.float64] | None = None
    #: Support pad task weight scales shaped (samples,).
    pad_weights: NDArray[np.float64] | None = None
    #: Posture references shaped (samples, configuration size), if not home.
    postures: NDArray[np.float64] | None = None
    #: Whether arms swing with the feet.
    arm_swing: bool = True
    #: Center of mass task weight scales per axis; a zero height leaves it to the solver.
    center_of_mass_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Knee bend targets of loaded legs in radians shaped (samples,), if any.
    knee_flexion: NDArray[np.float64] | None = None
    #: How firmly each leg holds its knee target, fading across contact changes (samples, 2).
    knee_activity: NDArray[np.float64] | None = None
    #: Weight of the stance knee targets.
    knee_weight: float = _TRACKING_KNEE_WEIGHT
    #: Arm swing in radians per meter of foot travel shaped (samples,), robot default if unset.
    arm_swing_gains: NDArray[np.float64] | None = None
    #: Forward offset of the arm swing center in radians shaped (samples,), if any.
    arm_biases: NDArray[np.float64] | None = None
    #: Largest arm swing either way in radians shaped (samples,), if any.
    arm_limits: NDArray[np.float64] | None = None
    #: Elbow bends in radians shaped (samples,), the home pose if unset.
    elbow_flexions: NDArray[np.float64] | None = None


def _knee_targets(
    robot_model: RobotModel, plan: MotionPlan, index: int
) -> dict[int, tuple[float, float]]:
    """Ask planted, unrolled legs for the planned knee bend."""
    if plan.knee_flexion is None or plan.knee_activity is None:
        return {}
    targets = {}
    for side, (name, sign) in enumerate(robot_model.robot.knee_flexion.items()):
        if (activity := float(plan.knee_activity[index, side])) > 0:
            address = int(robot_model.model.jnt_qposadr[robot_model.model.joint(name).id])
            weight = plan.knee_weight * activity
            targets[address] = (sign * float(plan.knee_flexion[index]), weight)
    return targets


def _knee_activity(feet: FootStates, interval: float, *, periodic: bool) -> NDArray[np.float64]:
    """Fade knee targets in after touchdown and out before liftoff or toe roll.

    Args:
        feet: Sole trajectories and contacts.
        interval: Planner grid spacing in seconds.
        periodic: Whether the samples wrap around.

    Returns:
        Knee target activity in [0, 1] shaped (samples, 2).
    """
    planted = (feet.contacts & (feet.pitches < 1e-3)).astype(np.float64)
    width = max(1, round(_KNEE_FADE / interval))
    window = np.hanning(2 * width + 1)
    window /= window.sum()
    activity = np.zeros_like(planted)
    for side in range(2):
        if periodic:
            padded = np.concatenate(
                (planted[-width:, side], planted[:, side], planted[:width, side])
            )
        else:
            padded = np.pad(planted[:, side], width, mode="edge")
        activity[:, side] = np.convolve(padded, window, mode="valid")
    # Only fade within the stance, never pull on a swinging knee.
    return activity * planted


def _arm_angles(robot_model: RobotModel, plan: MotionPlan, index: int) -> dict[int, float]:
    """Swing each arm opposite to the foot on its side and set the elbow bend."""
    robot = robot_model.robot
    if not plan.arm_swing or not robot_model.arm_swing:
        return {}
    heading = np.array([np.cos(plan.base_yaws[index]), np.sin(plan.base_yaws[index])])
    gain = robot.arm_swing_gain if plan.arm_swing_gains is None else plan.arm_swing_gains[index]
    bias = 0.0 if plan.arm_biases is None else float(plan.arm_biases[index])
    limit = _DEFAULT_ARM_LIMIT if plan.arm_limits is None else float(plan.arm_limits[index])
    angles = {}
    for side, (address, direction) in enumerate(robot_model.arm_swing):
        relative = plan.feet.positions[index, side, :2] - plan.center_of_mass.positions[index, :2]
        # Saturate smoothly so a strong swing rounds off instead of stopping abruptly.
        swing = limit * np.tanh(-gain * (relative @ heading) / limit) if limit > 0 else 0.0
        angles[address] = float(
            robot_model.home_configuration[address] + direction * (swing + bias)
        )
    if plan.elbow_flexions is not None:
        for name, sign in robot.elbow_flexion_signs.items():
            address = int(robot_model.model.jnt_qposadr[robot_model.model.joint(name).id])
            angles[address] = sign * float(plan.elbow_flexions[index])
    return angles


def _home_elbow_flexion(robot_model: RobotModel) -> float:
    """Return the mean elbow bend of the home pose in radians."""
    model = robot_model.model

    return (
        float(np.mean(bends))
        if (
            bends := [
                sign * robot_model.home_configuration[model.jnt_qposadr[model.joint(name).id]]
                for name, sign in robot_model.robot.elbow_flexion_signs.items()
            ]
        )
        else 0.0
    )


def _elbow_flexion(robot_model: RobotModel, parameters: GaitParameters) -> float:
    """Return the elbow bend of a gait, the home bend when it sets none."""
    if parameters.elbow_flexion is None:
        return _home_elbow_flexion(robot_model)
    return parameters.elbow_flexion


def _yaw_momentum(
    robot_model: RobotModel, configurations: NDArray[np.float64], interval: float
) -> NDArray[np.float64]:
    """Return the vertical angular momentum about the center of mass of inner frames."""
    model = robot_model.model
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    velocity = np.zeros(model.nv)
    momenta = []
    for index in range(1, len(configurations) - 1):
        mujoco.mj_differentiatePos(  # ty: ignore[unresolved-attribute]
            model, velocity, 2 * interval, configurations[index - 1], configurations[index + 1]
        )
        data.qpos[:] = configurations[index]
        data.qvel[:] = velocity
        mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
        mujoco.mj_comPos(model, data)  # ty: ignore[unresolved-attribute]
        mujoco.mj_comVel(model, data)  # ty: ignore[unresolved-attribute]
        mujoco.mj_subtreeVel(model, data)  # ty: ignore[unresolved-attribute]
        momenta.append(data.subtree_angmom[robot_model.root_body, 2])
    return np.asarray(momenta)


def balanced_arm_gain(robot_model: RobotModel, plan: MotionPlan) -> float:
    """Find the arm swing gain that best cancels the legs' vertical angular momentum.

    The momentum depends almost linearly on the swing gain, so solving the motion
    with the arms still and with a probe gain gives the cancelling gain directly.

    Args:
        robot_model: Robot to animate.
        plan: Plan whose motion is balanced.

    Returns:
        The arm swing gain in radians per meter of foot travel.
    """
    if not robot_model.arm_swing:
        return 0.0
    indices = np.arange(0, len(plan.times), PLANNER_SUBDIVISIONS)
    interval = float(plan.times[PLANNER_SUBDIVISIONS] - plan.times[0])
    probe = robot_model.robot.arm_swing_gain
    momenta = []
    for gain in (0.0, probe):
        trial = replace(plan, arm_swing_gains=np.full(len(plan.times), gain))
        configurations, _ = _solve(robot_model, trial, indices)
        momenta.append(_yaw_momentum(robot_model, configurations, interval))
    still, change = momenta[0], momenta[1] - momenta[0]
    scale = -float(still @ change) / max(float(change @ change), 1e-12)
    return float(np.clip(scale, 0.0, _ARM_GAIN_SCALE_LIMIT) * probe)


def solve_frames(
    robot_model: RobotModel, plan: MotionPlan, indices: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Solve whole-body configurations for selected planner samples.

    Args:
        robot_model: Robot to animate.
        plan: Task-space plan.
        indices: Planner sample indices to solve, in playback order.

    Returns:
        Configurations shaped (frames, configuration size).

    Raises:
        ReachError: If a frame cannot be reached.
    """
    configurations, misses = _solve(robot_model, plan, indices)
    if (worst := int(np.argmax(misses))) >= 0 and misses[worst] > 0:
        error = misses[worst]
        raise ReachError(float(error), float(plan.times[indices[worst]]))
    return configurations


def _solve(
    robot_model: RobotModel, plan: MotionPlan, indices: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve configurations and how far each frame overshoots the tracking tolerances.

    Returns:
        Configurations shaped (frames, size) and per-frame tracking errors in meters,
        zero for frames within tolerance.
    """
    solver = InverseKinematics(robot_model)
    configuration = solver.solve(standing_target(robot_model), robot_model.home_configuration)
    current = configuration.configuration
    robot = robot_model.robot
    frames, misses = [], []
    for index in indices:
        # Swing feet are released quickly after liftoff and caught just before touchdown.
        fraction = plan.feet.swing_fractions[index]
        release = np.where(
            fraction > 0, minimum_jerk(np.minimum(fraction, 1 - fraction) / _SWING_RELEASE), 0.0
        )
        # In flight nothing may disturb the ballistic center of mass.
        swing_weight = _SWING_POSITION_WEIGHT * (
            _FLIGHT_SWING_SCALE if not plan.feet.contacts[index].any() else 1.0
        )

        target = PoseTarget(
            sole_positions=plan.feet.positions[index],
            sole_yaws=plan.feet.yaws[index],
            center_of_mass=plan.center_of_mass.positions[index],
            base_quaternion=yaw_quaternion(plan.base_yaws[index], plan.base_pitches[index]),
            pad_position=None if plan.pad_positions is None else plan.pad_positions[index],
            pad_weight=1.0 if plan.pad_weights is None else float(plan.pad_weights[index]),
            center_of_mass_axes=plan.center_of_mass_axes,
            imposed_joints=_arm_angles(robot_model, plan, index),
            sole_orientation_weights=tuple(1.0 - (1.0 - _SWING_ORIENTATION_WEIGHT) * release),
            sole_position_weights=tuple(1.0 - (1.0 - swing_weight) * release),
            sole_pitches=tuple(plan.feet.pitches[index]),
            joint_targets=_knee_targets(robot_model, plan, index),
        )
        posture = None if plan.postures is None else plan.postures[index]
        result = solver.solve(target, current, posture)
        missed = (
            result.position_error > robot.position_tolerance
            or result.swing_error > robot.swing_tolerance
        )
        misses.append(max(result.position_error, result.swing_error) if missed else 0.0)
        current = result.configuration
        frames.append(current.copy())
    return np.asarray(frames), np.asarray(misses)


def _pad_heights(
    robot_model: RobotModel, configurations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute the support pad bottom height of configurations shaped (frames, size)."""
    model = robot_model.model
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    heights = np.zeros(len(configurations))
    for index, configuration in enumerate(configurations):
        data.qpos[:] = configuration
        mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
        heights[index] = data.site_xpos[robot_model.pad_site, 2]
    return heights


def solved_center_of_mass(
    robot_model: RobotModel, configurations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute the whole-body center of mass of configurations shaped (frames, size)."""
    model = robot_model.model
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    result = np.zeros((len(configurations), 3))
    for index, configuration in enumerate(configurations):
        data.qpos[:] = configuration
        mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
        mujoco.mj_comPos(model, data)  # ty: ignore[unresolved-attribute]
        result[index] = data.subtree_com[robot_model.root_body]
    return result


def build_animation(
    robot_model: RobotModel,
    plan: MotionPlan,
    *,
    name: str,
    kind: AnimationKind,
    sample_interval: float,
    indices: NDArray[np.int64],
    loop_offset: LoopOffset | None = None,
    parameters: dict[str, object] | None = None,
    origin: NDArray[np.float64] | None = None,
) -> Animation:
    """Solve the frames of a plan and package them as an animation.

    Args:
        robot_model: Robot to animate.
        plan: Task-space plan.
        name: Animation name.
        kind: Loop or action.
        sample_interval: Time between frames in seconds.
        indices: Planner sample index of every frame.
        loop_offset: Transform between loop cycles, identity when omitted.
        parameters: Generator settings to record.
        origin: Planar pose `[x, y, yaw]` moved to the world origin, if any.

    Returns:
        The animation.
    """
    configurations = solve_frames(robot_model, plan, indices)
    positions = configurations[:, :3].copy()
    quaternions = configurations[:, 3:7].copy()
    center_of_mass = plan.center_of_mass.positions[indices].copy()
    if plan.center_of_mass_axes[2] == 0:
        center_of_mass[:, 2] = solved_center_of_mass(robot_model, configurations)[:, 2]
    zero_moment_point = np.concatenate(
        (plan.center_of_mass.zero_moment_point[indices], np.zeros((len(indices), 1))), axis=1
    )
    if origin is not None:
        shift = LoopOffset(*origin).inverse()
        positions = shift.apply_positions(positions)
        quaternions = shift.apply_quaternions(quaternions)
        center_of_mass = shift.apply_positions(center_of_mass)
        zero_moment_point = shift.apply_positions(zero_moment_point)
    pad_contacts = plan.pad_contacts[indices]
    if plan.pad_positions is not None:
        pad_contacts = _pad_heights(robot_model, configurations) < _PAD_CONTACT_HEIGHT
    contacts = np.concatenate((plan.feet.contacts[indices], pad_contacts[:, None]), axis=1)
    return Animation(
        name=name,
        variant=str(robot_model.robot.variant),
        kind=kind,
        sample_interval=sample_interval,
        joint_names=robot_model.joint_names,
        root_positions=positions.astype(np.float32),
        root_quaternions=(quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)).astype(
            np.float32
        ),
        joint_positions=configurations[:, robot_model.joint_position_addresses].astype(np.float32),
        commands=plan.commands[indices].astype(np.float32),
        behaviors=tuple(plan.behaviors[index] for index in indices),
        phases=plan.phases[indices].astype(np.float32),
        contact_names=robot_model.robot.contact_names,
        contacts=contacts,
        center_of_mass=center_of_mass.astype(np.float32),
        zero_moment_point=zero_moment_point[:, :2].astype(np.float32),
        loop_offset=loop_offset or LoopOffset(),
        parameters=parameters or {},
        metadata={"generator": "open_stompers_animations"},
    )


def _quantize_cycle(parameters: GaitParameters, sample_interval: float) -> GaitParameters:
    """Round the two-step gait cycle to a whole number of frames."""
    frames = max(2, round(2 * parameters.step_period / sample_interval))
    return replace(parameters, cadence=120.0 / (frames * sample_interval))


def plan_loop(
    robot_model: RobotModel, parameters: GaitParameters, sample_interval: float
) -> tuple[MotionPlan, LoopOffset, GaitParameters]:
    """Plan one steady-state gait cycle.

    Args:
        robot_model: Robot to animate.
        parameters: Gait of the loop; its cadence is quantized to whole frames.
        sample_interval: Time between frames in seconds.

    Returns:
        The plan of one cycle, the transform between cycles, and the quantized gait.
    """
    parameters = _quantize_cycle(parameters, sample_interval)
    cycle = 2 * parameters.step_period
    interval = sample_interval / PLANNER_SUBDIVISIONS
    count = round(cycle / interval)
    times = np.arange(count, dtype=np.float64) * interval
    footsteps, path = plan_loop_footsteps(
        robot_model.robot, parameters, robot_model.stance_width / 2
    )
    feet = footsteps.evaluate(times, robot_model.sole_half_extents[:, 0])
    extended = np.arange(-count, 2 * count, dtype=np.float64) * interval
    reference = support_reference(
        extended,
        footsteps.evaluate(extended),
        parameters.step_period,
        robot_model.robot.support_inset,
    )[count : 2 * count]
    offset = LoopOffset(*path.pose(cycle)[0])
    base_yaws = path.pose(times)[:, 2]
    airborne = ~feet.contacts.any(axis=1)

    def center_of_mass(heights: NDArray[np.float64]) -> CenterOfMassPlan:
        return _center_of_mass_for_heights(
            interval, reference, heights, airborne, periodic=True, offset=offset
        )

    plan = MotionPlan(
        times=times,
        center_of_mass=center_of_mass(np.full(count, robot_model.center_of_mass_height)),
        feet=feet,
        base_yaws=base_yaws,
        base_pitches=np.full(count, parameters.base_pitch),
        commands=np.tile(parameters.command, (count, 1)),
        behaviors=(GAIT_BEHAVIORS[parameters.gait],) * count,
        phases=times / cycle,
        pad_contacts=np.zeros(count, dtype=bool),
        knee_flexion=np.full(count, parameters.stance_knee_flexion),
        knee_activity=_knee_activity(feet, interval, periodic=True),
        arm_biases=np.full(count, parameters.arm_forward_bias),
        arm_limits=np.full(count, parameters.arm_swing_limit),
        elbow_flexions=np.full(count, _elbow_flexion(robot_model, parameters)),
    )
    plan = _discover_heights(robot_model, plan, center_of_mass, cycle)
    gain = balanced_arm_gain(robot_model, plan)
    return replace(plan, arm_swing_gains=np.full(count, gain)), offset, parameters


def _segment_values(
    values: NDArray[np.float64],
    boundaries: NDArray[np.float64],
    ramps: NDArray[np.float64],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Blend per-segment values linearly over a ramp after each boundary."""
    result = np.full(len(times), values[0], dtype=np.float64)
    for index in range(1, len(values)):
        progress = np.clip((times - boundaries[index]) / ramps[index], 0, 1)
        result += progress * (values[index] - values[index - 1])
    return result


def _home_knee_flexion(robot_model: RobotModel) -> float:
    """Return the mean knee bend of the home pose in radians."""
    model = robot_model.model
    bends = [
        sign * robot_model.home_configuration[model.jnt_qposadr[model.joint(name).id]]
        for name, sign in robot_model.robot.knee_flexion.items()
    ]
    return float(np.mean(bends))


def _stepping_weights(
    footsteps: FootstepPlan, times: NDArray[np.float64], shift: float
) -> NDArray[np.float64]:
    """Return one while stepping, blending to zero while standing still."""
    weight = np.zeros(len(times))
    bouts: list[list[float]] = []
    for step in footsteps.steps:
        if bouts and step.lift_time - bouts[-1][1] < shift:
            bouts[-1][1] = max(bouts[-1][1], step.touchdown_time)
        else:
            bouts.append([step.lift_time, step.touchdown_time])
    for start, stop in bouts:
        rising = minimum_jerk((times - (start - shift)) / shift)
        falling = 1 - minimum_jerk((times - stop) / shift)
        weight = np.maximum(weight, np.minimum(rising, falling))
    return weight


def _center_of_mass_for_heights(
    interval: float,
    reference: NDArray[np.float64],
    heights: NDArray[np.float64],
    airborne: NDArray[np.bool_],
    *,
    periodic: bool,
    offset: LoopOffset | None = None,
) -> CenterOfMassPlan:
    """Plan a smooth center of mass that stays at or below the given stance heights.

    Args:
        interval: Planner grid spacing in seconds.
        reference: Zero moment point reference shaped (samples, 2).
        heights: Highest reachable stance heights shaped (samples,).
        airborne: Samples in free fall shaped (samples,).
        periodic: Whether the plan repeats.
        offset: Transform between cycles of a periodic plan.

    Returns:
        The center of mass plan.
    """
    weights = np.where(airborne, 0.0, 1.0)
    target = heights.copy()
    for _ in range(_LOWERING_ITERATIONS):
        vertical = plan_vertical(interval, target, weights, airborne, periodic=periodic)
        # Smoothing cuts corners, so push the target down where it overshoots.
        excess = np.where(airborne, 0.0, vertical[0] - heights)
        if excess.max() <= 1e-4:
            break
        target -= 1.5 * np.maximum(excess, 0.0)
    return plan_center_of_mass(interval, reference, *vertical, periodic=periodic, offset=offset)


def _discover_heights(
    robot_model: RobotModel,
    plan: MotionPlan,
    center_of_mass: Callable[[NDArray[np.float64]], CenterOfMassPlan],
    period: float | None = None,
) -> MotionPlan:
    """Let the solver choose the body height for the planned knee bends, then replan.

    With the center of mass height left free, every frame settles at the tallest pose
    whose planted knees bend as planned while all joint limits hold. The vertical and
    horizontal plans are then rebuilt around those heights, which the final solve
    tracks exactly.

    Args:
        robot_model: Robot to animate.
        plan: Plan with knee bend targets.
        center_of_mass: Planner turning stance heights (samples,) into a full plan.
        period: Cycle duration of periodic plans.

    Returns:
        The plan with discovered heights.
    """
    indices = np.arange(0, len(plan.times), PLANNER_SUBDIVISIONS)
    ground_feet = plan.feet
    for _ in range(_HEIGHT_PASSES):
        free = replace(
            plan, center_of_mass_axes=(1.0, 1.0, 0.0), knee_weight=_DISCOVERY_KNEE_WEIGHT
        )
        configurations, _ = _solve(robot_model, free, indices)
        solved = solved_center_of_mass(robot_model, configurations)[:, 2]
        heights = np.interp(plan.times, plan.times[indices], solved, period=period)
        planned = center_of_mass(heights)
        plan = replace(
            plan,
            center_of_mass=planned,
            feet=_lift_swing_feet(
                ground_feet, planned.positions[:, 2], periodic=period is not None
            ),
        )
    return plan


def _lift_swing_feet(
    feet: FootStates, heights: NDArray[np.float64], *, periodic: bool
) -> FootStates:
    """Raise swing feet by however much the body rises above their liftoff and landing.

    Swing clearance is planned relative to the ground, but a body that rises during
    the swing, as in a flight phase, carries the legs up with it.

    Args:
        feet: Sole trajectories with ground-relative swing clearance.
        heights: Planned center of mass heights shaped (samples,).
        periodic: Whether swings may wrap around the end of the samples.

    Returns:
        Sole trajectories with lifted swings.
    """
    positions = feet.positions.copy()
    count = len(heights)
    for side in range(2):
        swinging = feet.swing_fractions[:, side] > 0
        starts = np.flatnonzero(swinging & ~np.roll(swinging, 1))
        for start in starts:
            stop = start
            while swinging[stop % count] and stop - start < count:
                stop += 1
            span = np.arange(start, stop)
            if not periodic and (start == 0 or stop >= count):
                continue
            before, after = heights[(start - 1) % count], heights[stop % count]
            fraction = feet.swing_fractions[span % count, side]
            baseline = (1 - fraction) * before + fraction * after
            positions[span % count, side, 2] += np.maximum(heights[span % count] - baseline, 0.0)
    return replace(feet, positions=positions)


def _acceleration_lean(
    parameters: list[GaitParameters],
    boundaries: NDArray[np.float64],
    times: NDArray[np.float64],
    period: float,
) -> NDArray[np.float64]:
    """Lean into forward acceleration along the gravito-inertial direction `atan(a / g)`.

    Args:
        parameters: Resolved segment parameters.
        boundaries: Segment start times plus the end time.
        times: Planner sample times in seconds.
        period: Smoothing window in seconds, about one step.

    Returns:
        Extra forward lean in radians shaped (samples,).
    """
    speed = schedule_twist(parameters, boundaries, times)[:, 0]
    acceleration = np.gradient(speed, times)
    width = max(1, round(period / (times[1] - times[0])))
    window = np.hanning(2 * width + 1)
    smooth = np.convolve(np.pad(acceleration, width, mode="edge"), window / window.sum(), "valid")
    return np.arctan2(smooth, GRAVITY)


@cache
def _gait_arm_gain(variant: HardwareVariant, gait: Gait) -> float:
    """Fit the momentum-cancelling arm swing gain of a gait at its default speed."""
    robot_model = build_robot_model(variant)
    settings = robot_model.robot.gaits[Gait.WALK if gait is Gait.STAND else gait]
    if not robot_model.arm_swing:
        return 0.0
    requested = resolve_segment(
        robot_model.robot,
        CommandSegment(
            gait if gait is not Gait.STAND else Gait.WALK, 1.0, (settings.default_speed, 0.0, 0.0)
        ),
        robot_model.height,
    )
    plan, _, _ = plan_loop(robot_model, requested, DEFAULT_SAMPLE_INTERVAL)
    gains = plan.arm_swing_gains
    return 0.0 if gains is None else float(gains[0])


def plan_sequence(
    robot_model: RobotModel,
    segments: list[CommandSegment] | tuple[CommandSegment, ...],
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    lowered: tuple[float, ...] = (),
) -> MotionPlan:
    """Plan a command schedule that starts and ends standing still.

    Args:
        robot_model: Robot to animate.
        segments: Command segments in playback order.
        sample_interval: Time between frames in seconds.
        lowered: Times around which the center of mass is lowered to extend reach.

    Returns:
        The plan, long enough for the robot to come to rest.
    """
    robot = robot_model.robot
    parameters, boundaries = resolve_schedule(robot, segments, robot_model.height)
    path = integrate_schedule(parameters, boundaries, float(boundaries[-1]) + 30.0)
    footsteps = plan_sequence_footsteps(
        robot, parameters, boundaries, path, robot_model.stance_width / 2
    )
    shift = min(
        (values.step_period for values in parameters if values.moving),
        default=parameters[0].step_period,
    )
    last_touchdown = max((step.touchdown_time for step in footsteps.steps), default=0.0)
    end = max(float(boundaries[-1]), last_touchdown + shift + FINAL_HOLD)
    interval = sample_interval / PLANNER_SUBDIVISIONS
    frames = int(np.ceil(end / sample_interval - 1e-9)) + 1
    count = (frames - 1) * PLANNER_SUBDIVISIONS + 1
    times = np.arange(count, dtype=np.float64) * interval
    feet = footsteps.evaluate(times, robot_model.sole_half_extents[:, 0])
    reference = support_reference(times, feet, shift, robot.support_inset)
    ramps = RAMP_STEPS * np.array([values.step_period for values in parameters])
    lowering = np.zeros(count)
    for time in lowered:
        lowering += (
            _REACH_LOWERING
            * robot_model.center_of_mass_height
            * np.exp(-0.5 * ((times - time) / _REACH_LOWERING_WIDTH) ** 2)
        )
    base_yaws = path.pose(times)[:, 2]
    airborne = ~feet.contacts.any(axis=1)
    stepping = _stepping_weights(footsteps, times, shift)
    # While standing still the knees return to their home bend.
    flexion = stepping * _segment_values(
        np.array([values.stance_knee_flexion for values in parameters]), boundaries, ramps, times
    ) + (1 - stepping) * _home_knee_flexion(robot_model)

    def center_of_mass(heights: NDArray[np.float64]) -> CenterOfMassPlan:
        return _center_of_mass_for_heights(
            interval, reference, heights - lowering, airborne, periodic=False
        )

    segment_indices = np.clip(
        np.searchsorted(boundaries, times, side="right") - 1, 0, len(parameters) - 1
    )
    plan = MotionPlan(
        times=times,
        center_of_mass=center_of_mass(np.full(count, robot_model.center_of_mass_height)),
        feet=feet,
        base_yaws=base_yaws,
        base_pitches=_segment_values(
            np.array([values.base_pitch for values in parameters]), boundaries, ramps, times
        )
        + _acceleration_lean(parameters, boundaries, times, shift),
        commands=np.array([parameters[index].command for index in segment_indices]),
        behaviors=tuple(GAIT_BEHAVIORS[parameters[index].gait] for index in segment_indices),
        phases=footsteps.phase(times),
        pad_contacts=np.zeros(count, dtype=bool),
        knee_flexion=flexion,
        knee_activity=_knee_activity(feet, interval, periodic=False),
        arm_swing_gains=_segment_values(
            np.array(
                [_gait_arm_gain(robot_model.robot.variant, values.gait) for values in parameters]
            ),
            boundaries,
            ramps,
            times,
        ),
        arm_biases=_segment_values(
            np.array([values.arm_forward_bias for values in parameters]), boundaries, ramps, times
        ),
        arm_limits=_segment_values(
            np.array([values.arm_swing_limit for values in parameters]), boundaries, ramps, times
        ),
        elbow_flexions=_segment_values(
            np.array([_elbow_flexion(robot_model, values) for values in parameters]),
            boundaries,
            ramps,
            times,
        ),
    )
    return _discover_heights(robot_model, plan, center_of_mass)


def frame_indices(
    plan: MotionPlan, sample_interval: float, start: float = 0.0, stop: float | None = None
) -> NDArray[np.int64]:
    """Select the planner samples of every frame in a time window.

    Args:
        plan: Plan whose grid is sampled.
        sample_interval: Time between frames in seconds.
        start: First kept time in seconds, rounded to a frame.
        stop: Last kept time in seconds, the plan end when omitted.

    Returns:
        Planner sample indices of the kept frames.
    """
    last = len(plan.times) - 1
    first_frame = round(start / sample_interval)
    last_frame = last // PLANNER_SUBDIVISIONS if stop is None else round(stop / sample_interval)
    frames = np.arange(first_frame, last_frame + 1)
    return np.clip(frames * PLANNER_SUBDIVISIONS, 0, last)


def plan_origin(plan: MotionPlan, index: int) -> NDArray[np.float64]:
    """Return the planar center of mass pose `[x, y, yaw]` at a planner sample."""
    position = plan.center_of_mass.positions[index]
    return np.array([position[0], position[1], plan.base_yaws[index]])


__all__ = [
    "FINAL_HOLD",
    "PLANNER_SUBDIVISIONS",
    "MotionPlan",
    "ReachError",
    "build_animation",
    "frame_indices",
    "plan_loop",
    "plan_origin",
    "plan_sequence",
    "solve_frames",
]
