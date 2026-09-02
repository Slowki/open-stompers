"""Whole-body differential inverse kinematics on the MuJoCo model."""

from dataclasses import dataclass, field
from typing import Final

import mujoco
import numpy as np
from numpy.typing import NDArray

from .model import RobotModel

#: Largest knee bend against its flexion direction, in radians.
_KNEE_HYPEREXTENSION: Final = 0.15


def yaw_quaternion(yaw: float, pitch: float = 0.0) -> NDArray[np.float64]:
    """Build a WXYZ quaternion from a heading followed by a pitch about the new Y axis."""
    heading = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    lean = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, heading, lean)  # ty: ignore[unresolved-attribute]
    return result


@dataclass(frozen=True)
class TaskWeights:
    """Relative importance of the inverse kinematics tasks."""

    #: Sole position tracking.
    sole_position: float = 1000.0
    #: Sole orientation tracking.
    sole_orientation: float = 100.0
    #: Whole-body center of mass tracking.
    center_of_mass: float = 1000.0
    #: Base orientation tracking.
    base_orientation: float = 5.0
    #: Support pad position tracking, when requested.
    pad_position: float = 30.0
    #: Pull of the leg joints toward the posture reference.
    posture: float = 1e-4
    #: Levenberg-Marquardt damping.
    damping: float = 1e-6


@dataclass(frozen=True)
class PoseTarget:
    """Desired world-frame pose of the solved tasks at one instant."""

    #: Left and right sole centers shaped (2, 3).
    sole_positions: NDArray[np.float64]
    #: Left and right sole headings in radians shaped (2,).
    sole_yaws: NDArray[np.float64]
    #: Whole-body center of mass shaped (3,).
    center_of_mass: NDArray[np.float64]
    #: Base WXYZ orientation shaped (4,).
    base_quaternion: NDArray[np.float64]
    #: Support pad bottom center shaped (3,), or None to leave it free.
    pad_position: NDArray[np.float64] | None = None
    #: Scale of the support pad task weight.
    pad_weight: float = 1.0
    #: Per-axis scale of the center of mass task weight shaped (3,).
    center_of_mass_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Scale of the left and right sole orientation task weights.
    sole_orientation_weights: tuple[float, float] = (1.0, 1.0)
    #: Scale of the left and right sole position task weights; below one marks a swing foot.
    sole_position_weights: tuple[float, float] = (1.0, 1.0)
    #: Left and right toe-down sole pitches in radians.
    sole_pitches: tuple[float, float] = (0.0, 0.0)
    #: Soft leg joint targets as `(angle, weight)` keyed by position address.
    joint_targets: dict[int, tuple[float, float]] = field(default_factory=dict)
    #: Joint angles imposed directly, keyed by position address.
    imposed_joints: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveResult:
    """Inverse kinematics solution and its remaining error."""

    #: Generalized coordinates.
    configuration: NDArray[np.float64]
    #: Largest remaining stance sole or center of mass position error in meters.
    position_error: float
    #: Largest remaining swing sole position error in meters.
    swing_error: float
    #: Largest remaining sole orientation error in radians.
    orientation_error: float


class InverseKinematics:
    """Damped least-squares whole-body solver over the root and leg joints."""

    def __init__(
        self,
        robot_model: RobotModel,
        weights: TaskWeights | None = None,
        *,
        iterations: int = 100,
        tolerance: float = 1e-9,
        maximum_step: float = 0.1,
    ) -> None:
        """Prepare buffers for a robot model.

        Args:
            robot_model: Model to solve on.
            weights: Task weights, defaults when omitted.
            iterations: Largest number of Newton steps per solve.
            tolerance: Squared step norm below which a solve stops.
            maximum_step: Largest change of any coordinate per Newton step.
        """
        self.robot_model = robot_model
        self.weights = weights or TaskWeights()
        self.iterations = iterations
        self.tolerance = tolerance
        self.maximum_step = maximum_step
        model = robot_model.model
        self._data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
        self._columns = robot_model.solved_velocity_indices
        leg_joints = [model.joint(name).id for name in robot_model.robot.leg_joints]
        self._leg_positions = np.array([model.jnt_qposadr[joint] for joint in leg_joints])
        self._leg_velocities = np.array([model.jnt_dofadr[joint] for joint in leg_joints])
        self._leg_columns = np.array(
            [int(np.flatnonzero(self._columns == velocity)[0]) for velocity in self._leg_velocities]
        )
        self._leg_column_of = dict(
            zip(self._leg_positions.tolist(), self._leg_columns.tolist(), strict=True)
        )
        self._lower = model.jnt_range[leg_joints, 0].copy()
        self._upper = model.jnt_range[leg_joints, 1].copy()
        # Keep knees on their flexion branch instead of flipping through straight.
        for index, name in enumerate(robot_model.robot.leg_joints):
            match robot_model.robot.knee_flexion.get(name):
                case None:
                    pass
                case sign if sign > 0:
                    self._lower[index] = max(self._lower[index], -_KNEE_HYPEREXTENSION)
                case _:
                    self._upper[index] = min(self._upper[index], _KNEE_HYPEREXTENSION)
        self._translation = np.zeros((3, model.nv))
        self._axes = np.sqrt(np.asarray(robot_model.robot.base_orientation_axes))
        self._rotation = np.zeros((3, model.nv))

    def _rotation_error(self, target: NDArray, current: NDArray) -> NDArray:
        """Return the world-frame rotation vector taking `current` to `target`."""
        local = np.zeros(3)
        mujoco.mju_subQuat(local, target, current)  # ty: ignore[unresolved-attribute]
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, current)  # ty: ignore[unresolved-attribute]
        return matrix.reshape(3, 3) @ local

    def _tasks(
        self, target: PoseTarget
    ) -> tuple[list[tuple[NDArray, NDArray, float]], float, float, float]:
        """Evaluate task Jacobians, errors, and weights at the current configuration."""
        model, data, weights = self.robot_model.model, self._data, self.weights
        tasks, position_error, orientation_error, swing_error = [], 0.0, 0.0, 0.0
        for side, site in enumerate(self.robot_model.sole_sites):
            mujoco.mj_jacSite(model, data, self._translation, self._rotation, site)  # ty: ignore[unresolved-attribute]
            error = target.sole_positions[side] - data.site_xpos[site]
            current = np.zeros(4)
            mujoco.mju_mat2Quat(current, data.site_xmat[site])  # ty: ignore[unresolved-attribute]
            rotation = self._rotation_error(
                yaw_quaternion(target.sole_yaws[side], target.sole_pitches[side]), current
            )
            position_scale = target.sole_position_weights[side]
            tasks.append(
                (
                    self._translation[:, self._columns].copy(),
                    error,
                    weights.sole_position * position_scale,
                )
            )
            scale = target.sole_orientation_weights[side]
            tasks.append(
                (
                    self._rotation[:, self._columns].copy(),
                    rotation,
                    weights.sole_orientation * scale,
                )
            )
            if position_scale >= 1.0:
                position_error = max(position_error, float(np.abs(error).max()))
            else:
                swing_error = max(swing_error, float(np.abs(error).max()))
            if scale >= 1.0:
                orientation_error = max(orientation_error, float(np.abs(rotation).max()))
        root = self.robot_model.root_body
        mujoco.mj_jacSubtreeCom(model, data, self._translation, root)  # ty: ignore[unresolved-attribute]
        axes = np.sqrt(np.asarray(target.center_of_mass_axes))[:, None]
        error = target.center_of_mass - data.subtree_com[root]
        tasks.append(
            (
                axes * self._translation[:, self._columns],
                axes[:, 0] * error,
                weights.center_of_mass,
            )
        )
        strict = np.asarray(target.center_of_mass_axes) >= 1.0
        if strict.any():
            position_error = max(position_error, float(np.abs(error[strict]).max()))
        mujoco.mj_jacBody(model, data, None, self._rotation, root)  # ty: ignore[unresolved-attribute]
        rotation = self._rotation_error(target.base_quaternion, data.xquat[root])
        # Weight the base axes separately, so robots may roll or pitch freely.
        frame = data.xmat[root].reshape(3, 3)
        tasks.append(
            (
                self._axes[:, None] * (frame.T @ self._rotation[:, self._columns]),
                self._axes * (frame.T @ rotation),
                weights.base_orientation,
            )
        )
        if target.pad_position is not None:
            site = self.robot_model.pad_site
            mujoco.mj_jacSite(model, data, self._translation, None, site)  # ty: ignore[unresolved-attribute]
            error = target.pad_position - data.site_xpos[site]
            tasks.append(
                (
                    self._translation[:, self._columns].copy(),
                    error,
                    weights.pad_position * target.pad_weight,
                )
            )
        return tasks, position_error, orientation_error, swing_error

    def _limited_step(
        self, hessian: NDArray, gradient: NDArray, configuration: NDArray
    ) -> NDArray[np.float64]:
        """Solve the Newton step, freezing leg joints that would push past a limit."""
        angles = configuration[self._leg_positions]
        free = np.ones(len(gradient), dtype=bool)
        for _ in range(len(self._leg_columns)):
            solution = np.zeros(len(gradient))
            solution[free] = np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
            steps = solution[self._leg_columns]
            blocked = ((angles <= self._lower + 1e-9) & (steps < 0)) | (
                (angles >= self._upper - 1e-9) & (steps > 0)
            )
            blocked &= free[self._leg_columns]
            if not blocked.any():
                return solution
            free[self._leg_columns[blocked]] = False
        return solution

    def solve(
        self,
        target: PoseTarget,
        initial: NDArray[np.float64],
        posture: NDArray[np.float64] | None = None,
    ) -> SolveResult:
        """Solve for a configuration reaching a pose target.

        Args:
            target: Desired task poses.
            initial: Starting generalized coordinates, typically the previous frame.
            posture: Configuration whose leg angles regularize the solution;
                the robot home configuration when omitted.

        Returns:
            The solved configuration and its remaining task errors.
        """
        model, data, weights = self.robot_model.model, self._data, self.weights
        reference = self.robot_model.home_configuration if posture is None else posture
        configuration = np.array(initial, dtype=np.float64, copy=True)
        for address, angle in target.imposed_joints.items():
            configuration[address] = angle
        step = np.zeros(model.nv)
        for _ in range(self.iterations):
            data.qpos[:] = configuration
            mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
            mujoco.mj_comPos(model, data)  # ty: ignore[unresolved-attribute]
            tasks, _, _, _ = self._tasks(target)
            size = len(self._columns)
            hessian = np.eye(size) * weights.damping
            gradient = np.zeros(size)
            for jacobian, error, weight in tasks:
                hessian += weight * jacobian.T @ jacobian
                gradient += weight * jacobian.T @ error
            hessian[self._leg_columns, self._leg_columns] += weights.posture
            gradient[self._leg_columns] += weights.posture * (
                reference[self._leg_positions] - configuration[self._leg_positions]
            )
            for address, (angle, weight) in target.joint_targets.items():
                column = self._leg_column_of[address]
                hessian[column, column] += weight
                gradient[column] += weight * (angle - configuration[address])
            solution = self._limited_step(hessian, gradient, configuration)
            if (largest := np.abs(solution).max()) > self.maximum_step:
                solution *= self.maximum_step / largest
            step[:] = 0
            step[self._columns] = solution
            mujoco.mj_integratePos(model, configuration, step, 1.0)  # ty: ignore[unresolved-attribute]
            configuration[self._leg_positions] = np.clip(
                configuration[self._leg_positions], self._lower, self._upper
            )
            if float(solution @ solution) < self.tolerance:
                break
        data.qpos[:] = configuration
        mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
        mujoco.mj_comPos(model, data)  # ty: ignore[unresolved-attribute]
        _, position_error, orientation_error, swing_error = self._tasks(target)
        return SolveResult(configuration, position_error, swing_error, orientation_error)


def standing_target(
    robot_model: RobotModel,
    *,
    height_offset: float = 0.0,
    imposed_joints: dict[int, float] | None = None,
) -> PoseTarget:
    """Build the target of a level stance centered on the origin.

    Args:
        robot_model: Robot to stand.
        height_offset: Center of mass height change from home, in meters.
        imposed_joints: Joint angles imposed directly, keyed by position address.

    Returns:
        A target with both soles flat and the center of mass above their midpoint.
    """
    half_width = robot_model.stance_width / 2
    return PoseTarget(
        sole_positions=np.array([[0.0, half_width, 0.0], [0.0, -half_width, 0.0]]),
        sole_yaws=np.zeros(2),
        center_of_mass=np.array([0.0, 0.0, robot_model.center_of_mass_height + height_offset]),
        base_quaternion=yaw_quaternion(0.0),
        imposed_joints=imposed_joints or {},
    )


__all__ = [
    "InverseKinematics",
    "PoseTarget",
    "SolveResult",
    "TaskWeights",
    "standing_target",
    "yaw_quaternion",
]
