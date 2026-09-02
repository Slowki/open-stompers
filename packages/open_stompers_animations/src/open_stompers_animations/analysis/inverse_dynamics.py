"""MuJoCo inverse dynamics of animations with contact force distribution.

For every frame the generalized forces that reproduce the animated motion are
computed with `mj_inverse` (contacts disabled). The six unactuated root forces
must come from the ground, so they are distributed over the planned contact
points inside linearized friction cones with non-negative least squares. What
the contacts cannot supply remains as a base residual, which measures how far
the animation is from being dynamically feasible. Joint torques are what is
left after removing the contact forces.
"""

from copy import copy
from dataclasses import dataclass
from typing import Final

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from ..model import RobotModel, build_robot_model
from ..robots import parse_variant
from ..sampling import sample_animation
from ..types import Animation, AnimationKind

#: Default Coulomb friction coefficient of the ground.
DEFAULT_FRICTION: Final = 0.8
_PYRAMID_EDGES: Final = 4
#: Corner height above which a foot rolling onto its toe no longer loads it, in meters.
_ROLLED_HEIGHT: Final = 0.003


@dataclass(frozen=True)
class InverseDynamics:
    """Per-frame forces required to perform an animation."""

    #: Frame times in seconds shaped (frames,).
    times: NDArray[np.float64]
    #: Joint names matching the joint columns.
    joint_names: tuple[str, ...]
    #: Contact body names matching the contact columns.
    contact_names: tuple[str, ...]
    #: Joint velocities in radians per second shaped (frames, joints).
    joint_velocities: NDArray[np.float64]
    #: Required joint torques in newton meters shaped (frames, joints).
    joint_torques: NDArray[np.float64]
    #: World-frame contact force on each contact body in newtons shaped (frames, contacts, 3).
    contact_forces: NDArray[np.float64]
    #: Center of pressure on the ground shaped (frames, 2), NaN while airborne.
    center_of_pressure: NDArray[np.float64]
    #: Signed distance of the center of pressure inside the support polygon in meters.
    support_margin: NDArray[np.float64]
    #: Root force (N) and torque (N m) the contacts cannot supply shaped (frames, 6).
    base_residual: NDArray[np.float64]
    #: Whole-body center of mass shaped (frames, 3).
    center_of_mass: NDArray[np.float64]
    #: Left and right sole centers shaped (frames, 2, 3).
    sole_positions: NDArray[np.float64]
    #: Slowest sole corner speed of feet planted for three frames, in m/s, shaped (frames, 2).
    foot_slip: NDArray[np.float64]
    #: Total joint mechanical power in watts shaped (frames,).
    mechanical_power: NDArray[np.float64]
    #: Deepest self-collision penetration in meters shaped (frames,).
    self_collision_depth: NDArray[np.float64]


def _configurations(robot_model: RobotModel, animation: Animation, times: NDArray) -> NDArray:
    """Assemble generalized coordinates of the animation at the given times."""
    sample = sample_animation(animation, times)
    configurations = np.tile(robot_model.home_configuration, (len(times), 1))
    configurations[:, :3] = sample.root_positions
    configurations[:, 3:7] = sample.root_quaternions
    for column, name in enumerate(animation.joint_names):
        address = robot_model.joint_position_addresses[robot_model.joint_index(name)]
        configurations[:, address] = sample.joint_positions[:, column]
    return configurations


def _padded_configurations(robot_model: RobotModel, animation: Animation) -> NDArray:
    """Return configurations with one extra frame on each side for finite differences."""
    model, interval = robot_model.model, animation.sample_interval
    if animation.kind is AnimationKind.LOOP:
        times = np.arange(-1, animation.frame_count + 1) * interval
        return _configurations(robot_model, animation, times)
    inner = _configurations(robot_model, animation, animation.times)
    padded = np.concatenate((inner[:1], inner, inner[-1:]))
    # Extrapolate with constant velocity so the end accelerations stay finite.
    for edge, neighbour, target in ((1, 2, 0), (-2, -3, -1)):
        velocity = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, velocity, 1.0, padded[neighbour], padded[edge])  # ty: ignore[unresolved-attribute]
        mujoco.mj_integratePos(model, padded[target], velocity, 1.0)  # ty: ignore[unresolved-attribute]
    return padded


def _rates(robot_model: RobotModel, padded: NDArray, interval: float) -> tuple[NDArray, NDArray]:
    """Compute central-difference generalized velocities and accelerations."""
    model = robot_model.model
    count = len(padded) - 2
    velocities, accelerations = np.zeros((count, model.nv)), np.zeros((count, model.nv))
    backward, forward = np.zeros(model.nv), np.zeros(model.nv)
    for index in range(count):
        mujoco.mj_differentiatePos(model, backward, interval, padded[index], padded[index + 1])  # ty: ignore[unresolved-attribute]
        mujoco.mj_differentiatePos(model, forward, interval, padded[index + 1], padded[index + 2])  # ty: ignore[unresolved-attribute]
        velocities[index] = (backward + forward) / 2
        accelerations[index] = (forward - backward) / interval
    return velocities, accelerations


def _contact_points(
    robot_model: RobotModel,
    data: mujoco.MjData,  # ty: ignore[unresolved-attribute]
    contacts: NDArray[np.bool_],
) -> list[tuple[int, int, NDArray]]:
    """List the `(contact index, body, world point)` sole and pad corners allowed to bear load."""
    sites = [*robot_model.sole_sites, robot_model.pad_site]
    pad = robot_model.robot.support_pad.half_size
    extents = [*robot_model.sole_half_extents, np.array(pad[:2])]
    points = []
    for index, (site, extent) in enumerate(zip(sites, extents, strict=True)):
        if not contacts[index]:
            continue
        rotation = data.site_xmat[site].reshape(3, 3)
        for signs in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            local = np.array([signs[0] * extent[0], signs[1] * extent[1], 0.0])
            point = data.site_xpos[site] + rotation @ local
            points.append((index, int(robot_model.model.site_bodyid[site]), point))
    # A foot rolling onto its toe only loads the corners still on the ground.
    grounded = [entry for entry in points if entry[2][2] < _ROLLED_HEIGHT]
    lowest = min((entry[2][2] for entry in points), default=0.0)
    return grounded if grounded and lowest < _ROLLED_HEIGHT else points


def _friction_pyramid(friction: float) -> NDArray[np.float64]:
    """Return linearized friction cone edges shaped (3, edges)."""
    angles = np.arange(_PYRAMID_EDGES) * 2 * np.pi / _PYRAMID_EDGES
    edges = np.stack((friction * np.cos(angles), friction * np.sin(angles), np.ones_like(angles)))
    return edges / np.linalg.norm(edges, axis=0)


def _convex_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the counter-clockwise convex hull of planar points."""
    ordered = sorted({(round(x, 9), round(y, 9)) for x, y in points})
    if len(ordered) < 3:
        return np.asarray(ordered)

    def chain(sequence: list[tuple[float, float]]) -> list[tuple[float, float]]:
        hull: list[tuple[float, float]] = []
        for point in sequence:
            while len(hull) >= 2:
                (ax, ay), (bx, by) = hull[-2], hull[-1]
                if (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) > 0:
                    break
                hull.pop()
            hull.append(point)
        return hull[:-1]

    return np.asarray(chain(ordered) + chain(ordered[::-1]))


def support_margin(point: NDArray[np.float64], polygon: NDArray[np.float64]) -> float:
    """Return the signed distance of a point inside a convex polygon, negative outside.

    Args:
        point: Planar point shaped (2,).
        polygon: Counter-clockwise polygon vertices shaped (vertices, 2).

    Returns:
        Distance to the nearest edge, positive when inside.
    """
    if len(polygon) < 3:
        if len(polygon) == 0:
            return float("nan")
        segment = polygon if len(polygon) == 2 else np.repeat(polygon, 2, axis=0)
        direction = segment[1] - segment[0]
        length = float(direction @ direction)
        fraction = 0.0 if length == 0 else np.clip((point - segment[0]) @ direction / length, 0, 1)
        return -float(np.linalg.norm(point - (segment[0] + fraction * direction)))
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack((-edges[:, 1], edges[:, 0]), axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return float(((point - polygon) * normals).sum(axis=1).min())


def _self_collision_model(
    robot_model: RobotModel,
) -> tuple[mujoco.MjModel, set[tuple[int, int]]]:  # ty: ignore[unresolved-attribute]
    """Copy the model for self-collision checks and find pairs touching at home."""
    model = copy(robot_model.model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")  # ty: ignore[unresolved-attribute]
    model.geom_contype[floor] = model.geom_conaffinity[floor] = 0
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    data.qpos[:] = robot_model.home_configuration
    mujoco.mj_fwdPosition(model, data)  # ty: ignore[unresolved-attribute]
    return model, _contact_pairs(data)


def _contact_pairs(data: mujoco.MjData) -> set[tuple[int, int]]:  # ty: ignore[unresolved-attribute]
    """Return the sorted geom pairs of the active contacts."""
    return {(int(min(pair)), int(max(pair))) for pair in data.contact.geom[: data.ncon]}


def _neighbour_contacts(
    contacts: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return contacts allowed to load and contacts held over three frames."""
    # 50 Hz frames cannot resolve touchdown instants, so neighbouring frames may load too.
    loadable, settled = contacts.copy(), contacts.copy()
    loadable[1:] |= contacts[:-1]
    loadable[:-1] |= contacts[1:]
    settled[1:] &= contacts[:-1]
    settled[:-1] &= contacts[1:]
    return loadable, settled


def _sole_geometry(
    robot_model: RobotModel,
    data: mujoco.MjData,  # ty: ignore[unresolved-attribute]
    configurations: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return sole centers (frames, 2, 3) and sole corners (frames, 2, 4, 3)."""
    model = robot_model.model
    centers = np.zeros((len(configurations), 2, 3))
    corners = np.zeros((len(configurations), 2, 4, 3))
    signs = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]], dtype=np.float64)
    for index, configuration in enumerate(configurations):
        data.qpos[:] = configuration
        mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
        for side, site in enumerate(robot_model.sole_sites):
            centers[index, side] = data.site_xpos[site]
            extent = np.array([*robot_model.sole_half_extents[side], 0.0])
            rotation = data.site_xmat[site].reshape(3, 3)
            corners[index, side] = data.site_xpos[site] + (signs * extent) @ rotation.T
    return centers, corners


def _distribute(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    data: mujoco.MjData,  # ty: ignore[unresolved-attribute]
    points: list[tuple[int, int, NDArray]],
    required: NDArray[np.float64],
    pyramid: NDArray[np.float64],
    regularization: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find non-negative friction cone forces best supplying the root wrench.

    Returns:
        The generalized contact force (nv,) and the world force of each point (points, 3).
    """
    jacobian = np.zeros((3, model.nv))
    columns = []
    for _, body, point in points:
        mujoco.mj_jac(model, data, jacobian, None, point, body)  # ty: ignore[unresolved-attribute]
        columns.append(jacobian.T @ pyramid)
    basis = np.concatenate(columns, axis=1)
    system = np.concatenate((basis[:6], np.sqrt(regularization) * np.eye(basis.shape[1])))
    target = np.concatenate((required[:6], np.zeros(basis.shape[1])))
    weights, _ = nnls(system, target, maxiter=50 * basis.shape[1])
    point_forces = (pyramid @ weights.reshape(len(points), _PYRAMID_EDGES).T).T
    return basis @ weights, point_forces


def _self_collision_depth(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    data: mujoco.MjData,  # ty: ignore[unresolved-attribute]
    configuration: NDArray[np.float64],
    touching: set[tuple[int, int]],
) -> float:
    """Return the deepest penetration between geoms that do not touch at home."""
    data.qpos[:] = configuration
    mujoco.mj_fwdPosition(model, data)  # ty: ignore[unresolved-attribute]
    pairs = data.contact.geom[: data.ncon]
    distances = data.contact.dist[: data.ncon]
    return max(
        (
            -float(distance)
            for pair, distance in zip(pairs, distances, strict=True)
            if (int(min(pair)), int(max(pair))) not in touching
        ),
        default=0.0,
    )


def inverse_dynamics(
    animation: Animation,
    *,
    friction: float = DEFAULT_FRICTION,
    regularization: float = 1e-8,
) -> InverseDynamics:
    """Compute the joint torques and contact forces an animation requires.

    Args:
        animation: Animation to analyze.
        friction: Ground Coulomb friction coefficient.
        regularization: Weight preferring evenly spread contact forces.

    Returns:
        Per-frame torques, contact forces, and feasibility measures.
    """
    robot_model = build_robot_model(parse_variant(animation.variant))
    model = copy(robot_model.model)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # ty: ignore[unresolved-attribute]
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    collision_model, touching = _self_collision_model(robot_model)
    collision_data = mujoco.MjData(collision_model)  # ty: ignore[unresolved-attribute]
    interval = animation.sample_interval
    padded = _padded_configurations(robot_model, animation)
    velocities, accelerations = _rates(robot_model, padded, interval)
    pyramid = _friction_pyramid(friction)
    count = animation.frame_count
    velocity_columns = np.array(
        [
            robot_model.joint_velocity_addresses[robot_model.joint_index(name)]
            for name in animation.joint_names
        ]
    )
    generalized = np.zeros((count, model.nv))
    forces = np.zeros((count, len(animation.contact_names), 3))
    pressure = np.full((count, 2), np.nan)
    margins = np.full(count, np.nan)
    center_of_mass = np.zeros((count, 3))
    depths = np.zeros(count)
    loadable, settled = _neighbour_contacts(animation.contacts)
    sole_positions, corners = _sole_geometry(robot_model, data, padded)
    for index in range(count):
        data.qpos[:] = padded[index + 1]
        data.qvel[:] = velocities[index]
        data.qacc[:] = accelerations[index]
        mujoco.mj_inverse(model, data)  # ty: ignore[unresolved-attribute]
        generalized[index] = data.qfrc_inverse
        center_of_mass[index] = data.subtree_com[robot_model.root_body]
        if points := _contact_points(robot_model, data, loadable[index]):
            supplied, point_forces = _distribute(
                model, data, points, generalized[index], pyramid, regularization
            )
            generalized[index] -= supplied
            for (contact, _, _), force in zip(points, point_forces, strict=True):
                forces[index, contact] += force
            locations = np.array([point for _, _, point in points])
            if (normal := point_forces[:, 2].sum()) > 1e-6:
                pressure[index] = (locations[:, :2] * point_forces[:, 2:3]).sum(axis=0) / normal
                margins[index] = support_margin(pressure[index], _convex_hull(locations[:, :2]))
        depths[index] = _self_collision_depth(
            collision_model, collision_data, padded[index + 1], touching
        )
    # A rolling foot keeps its pivot corner still, so take the slowest corner.
    corner_motion = corners[2:, ..., :2] - corners[:-2, ..., :2]
    sole_speeds = np.linalg.norm(corner_motion, axis=3).min(axis=2) / (2 * interval)
    torques = generalized[:, velocity_columns]
    joint_velocities = velocities[:, velocity_columns]
    return InverseDynamics(
        times=animation.times,
        joint_names=animation.joint_names,
        contact_names=animation.contact_names,
        joint_velocities=joint_velocities,
        joint_torques=torques,
        contact_forces=forces,
        center_of_pressure=pressure,
        support_margin=margins,
        base_residual=generalized[:, :6],
        center_of_mass=center_of_mass,
        sole_positions=sole_positions[1:-1],
        foot_slip=np.where(settled[:, :2], sole_speeds, 0.0),
        mechanical_power=(torques * joint_velocities).sum(axis=1),
        self_collision_depth=np.maximum(depths, 0.0),
    )


__all__ = [
    "DEFAULT_FRICTION",
    "InverseDynamics",
    "inverse_dynamics",
    "support_margin",
]
