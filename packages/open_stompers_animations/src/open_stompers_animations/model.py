"""MuJoCo models of the hardware variants with sole and support pad frames."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import mujoco
import numpy as np
from numpy.typing import NDArray

from open_stompers_hardware_models import (
    HardwareVariant,
    MicroduckVariant,
    load,
    microduck,
    mini_pi_plus,
    mjcf_metadata_for,
)
from open_stompers_hardware_models.mjcf import export_mjcf
from open_stompers_hardware_models.poses import home_pose

from .robots import ROBOTS, AnimationRobot

#: Height band above the lowest foot point that counts as sole surface, in meters.
SOLE_TOLERANCE: Final = 0.002
#: Name of the free joint added to robots exported without one.
ROOT_JOINT: Final = "root"
#: Sole frame site names, left then right.
SOLE_SITES: Final = ("left_sole", "right_sole")
#: Support pad bottom frame site name.
PAD_SITE: Final = "support_pad_bottom"

_CIRCLE_SAMPLES: Final = 24


@dataclass(frozen=True)
class RobotModel:
    """Compiled MuJoCo model plus the frames animation planning relies on."""

    #: Animation settings of the robot.
    robot: AnimationRobot
    #: Compiled model with a free root joint, sole sites, and a floor.
    model: mujoco.MjModel  # ty: ignore[unresolved-attribute]
    #: Hinge joint names in generalized coordinate order.
    joint_names: tuple[str, ...]
    #: Position address of every hinge joint shaped (joints,).
    joint_position_addresses: NDArray[np.int64]
    #: Velocity address of every hinge joint shaped (joints,).
    joint_velocity_addresses: NDArray[np.int64]
    #: Root body identifier.
    root_body: int
    #: Left and right sole site identifiers.
    sole_sites: tuple[int, int]
    #: Sole half length and half width shaped (2, 2), left then right.
    sole_half_extents: NDArray[np.float64]
    #: Support pad bottom site identifier.
    pad_site: int
    #: Crouched home joint angles shaped (joints,).
    home_joint_positions: NDArray[np.float64]
    #: Home configuration with level base and soles resting on the floor.
    home_configuration: NDArray[np.float64]
    #: Lateral distance between the sole centers at home, in meters.
    stance_width: float
    #: Center of mass height above the floor at home, in meters.
    center_of_mass_height: float
    #: Standing height from the soles to the top of the robot at home, in meters.
    height: float
    #: Total mass in kilograms.
    mass: float
    #: Joint lower limits in radians shaped (joints,).
    lower_limits: NDArray[np.float64]
    #: Joint upper limits in radians shaped (joints,).
    upper_limits: NDArray[np.float64]
    #: Joint effort limits in newton meters shaped (joints,).
    effort_limits: NDArray[np.float64]
    #: Joint speed limits in radians per second shaped (joints,).
    velocity_limits: NDArray[np.float64]
    #: Velocity indices of the root and leg joints solved by inverse kinematics.
    solved_velocity_indices: NDArray[np.int64]
    #: Left and right arm swing position addresses and forward directions.
    arm_swing: tuple[tuple[int, float], ...]

    def joint_index(self, name: str) -> int:
        """Return the column of a joint in animation joint arrays."""
        return self.joint_names.index(name)


def _ensure_meshes(variant: HardwareVariant) -> None:
    """Download the meshes of a variant into the hardware model cache."""
    match variant:
        case MicroduckVariant():
            microduck.ensure_meshes(variant)
        case _:
            mini_pi_plus.ensure_meshes(variant)


def _collision_points(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    data: mujoco.MjData,  # ty: ignore[unresolved-attribute]
    body: int,
) -> NDArray[np.float64]:
    """Return world points sampling the collision surfaces of a body."""
    points = []
    angles = np.linspace(0, 2 * np.pi, _CIRCLE_SAMPLES, endpoint=False)
    circle = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
    for index in np.flatnonzero(model.geom_bodyid == body):
        if not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        rotation = data.geom_xmat[index].reshape(3, 3)
        size = model.geom_size[index]
        match model.geom_type[index]:
            case mujoco.mjtGeom.mjGEOM_BOX | mujoco.mjtGeom.mjGEOM_ELLIPSOID:  # ty: ignore[unresolved-attribute]
                signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * 3))).reshape(3, -1).T
                local = signs * size
            case mujoco.mjtGeom.mjGEOM_CYLINDER:  # ty: ignore[unresolved-attribute]
                rim = circle * size[0]
                cap = np.array([0.0, 0.0, size[1]])
                local = np.concatenate((rim + cap, rim - cap))
            case mujoco.mjtGeom.mjGEOM_SPHERE:  # ty: ignore[unresolved-attribute]
                points.append(data.geom_xpos[index] - [0, 0, size[0]])
                continue
            case mujoco.mjtGeom.mjGEOM_CAPSULE:  # ty: ignore[unresolved-attribute]
                ends = np.array([[0, 0, size[1]], [0, 0, -size[1]]])
                world = ends @ rotation.T + data.geom_xpos[index]
                points.extend(world - [0, 0, size[0]])
                continue
            case mujoco.mjtGeom.mjGEOM_MESH:  # ty: ignore[unresolved-attribute]
                mesh = model.geom_dataid[index]
                start = model.mesh_vertadr[mesh]
                local = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            case _:
                continue
        points.extend(local @ rotation.T + data.geom_xpos[index])
    if not points:
        raise ValueError(f"Body '{model.body(body).name}' has no collision geometry")
    return np.asarray(points)


def _home_qpos(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    pose: dict[str, float],
) -> NDArray[np.float64]:
    """Build a configuration with the home joint angles and an identity root."""
    qpos = model.qpos0.copy()
    qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    for name, angle in pose.items():
        if (joint := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) >= 0:  # ty: ignore[unresolved-attribute]
            qpos[model.jnt_qposadr[joint]] = angle
    return qpos


def _build_spec(robot: AnimationRobot, directory: Path) -> mujoco.MjSpec:  # ty: ignore[unresolved-attribute]
    """Export the hardware model and add a free root, floor, and support pad."""
    _ensure_meshes(robot.variant)
    path = export_mjcf(
        load(robot.variant),
        directory / "robot.xml",
        mjcf_metadata=mjcf_metadata_for(robot.variant),
    )
    spec = mujoco.MjSpec.from_file(str(path))  # ty: ignore[unresolved-attribute]
    root = spec.worldbody.first_body()
    if not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints):  # ty: ignore[unresolved-attribute]
        root.add_freejoint(name=ROOT_JOINT)
    root.pos = (0.0, 0.0, 0.0)
    root.quat = (1.0, 0.0, 0.0, 0.0)
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,  # ty: ignore[unresolved-attribute]
        size=(5, 5, 0.1),
        rgba=(0.25, 0.3, 0.35, 1),
    )
    pad = robot.support_pad
    spec.body(pad.body).add_geom(
        name="support_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        pos=pad.position,
        size=pad.half_size,
        contype=0,
        conaffinity=0,
        group=3,
    )
    spec.body(pad.body).add_site(
        name=PAD_SITE,
        pos=(pad.position[0], pad.position[1], pad.position[2] - pad.half_size[2]),
    )
    return spec


def _sole_frames(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    robot: AnimationRobot,
    qpos: NDArray[np.float64],
) -> tuple[list[tuple[NDArray, NDArray]], NDArray[np.float64]]:
    """Find level sole frames in each foot body frame at the home pose."""
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
    frames, extents = [], []
    for name in robot.feet:
        body = model.body(name).id
        points = _collision_points(model, data, body)
        sole = points[points[:, 2] < points[:, 2].min() + SOLE_TOLERANCE]
        minimum, maximum = sole[:, :2].min(axis=0), sole[:, :2].max(axis=0)
        center = np.array([*(minimum + maximum) / 2, points[:, 2].min()])
        rotation = data.xmat[body].reshape(3, 3)
        local_position = rotation.T @ (center - data.xpos[body])
        local_quaternion = np.zeros(4)
        mujoco.mju_negQuat(local_quaternion, data.xquat[body])  # ty: ignore[unresolved-attribute]
        frames.append((local_position, local_quaternion))
        extents.append((maximum - minimum) / 2)
    return frames, np.asarray(extents)


def _standing_height(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    data: mujoco.MjData,  # ty: ignore[unresolved-attribute]
) -> float:
    """Return the highest point of any robot geometry, soles resting on the floor."""
    top = 0.0
    for index in np.flatnonzero(model.geom_bodyid > 0):
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:  # ty: ignore[unresolved-attribute]
            mesh = model.geom_dataid[index]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            rotation = data.geom_xmat[index].reshape(3, 3)
            highest = float((vertices @ rotation.T + data.geom_xpos[index])[:, 2].max())
        else:
            highest = float(data.geom_xpos[index, 2] + model.geom_rbound[index])
        top = max(top, highest)
    return top


def _arm_swing(
    model: mujoco.MjModel,  # ty: ignore[unresolved-attribute]
    robot: AnimationRobot,
    qpos: NDArray[np.float64],
) -> tuple[tuple[int, float], ...]:
    """Find the joint direction that swings each arm forward."""
    if robot.arm_swing_joints is None:
        return ()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    swing = []
    for name in robot.arm_swing_joints:
        joint = model.joint(name).id
        address, body = int(model.jnt_qposadr[joint]), model.jnt_bodyid[joint]
        positions = []
        for delta in (-0.1, 0.1):
            data.qpos[:] = qpos
            data.qpos[address] += delta
            mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
            mujoco.mj_comPos(model, data)  # ty: ignore[unresolved-attribute]
            positions.append(data.subtree_com[body, 0])
        swing.append((address, 1.0 if positions[1] > positions[0] else -1.0))
    return tuple(swing)


@cache
def build_robot_model(variant: HardwareVariant) -> RobotModel:
    """Compile the animation model of a hardware variant.

    Args:
        variant: Hardware variant to build.

    Returns:
        The compiled model with sole and pad frames.

    Raises:
        KeyError: If the variant has no animation settings.
    """
    robot = ROBOTS[variant]
    pose = home_pose(variant)
    with TemporaryDirectory(prefix="open-stompers-animations-") as directory:
        spec = _build_spec(robot, Path(directory))
        draft = spec.compile()
        frames, extents = _sole_frames(draft, robot, _home_qpos(draft, pose))
        for name, foot, (position, quaternion) in zip(SOLE_SITES, robot.feet, frames, strict=True):
            spec.body(foot).add_site(name=name, pos=position, quat=quaternion)
        model = spec.compile()
    hinges = [
        joint
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE  # ty: ignore[unresolved-attribute]
    ]
    joint_names = tuple(model.joint(joint).name for joint in hinges)
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    qpos = _home_qpos(model, pose)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
    sole_sites = (model.site(SOLE_SITES[0]).id, model.site(SOLE_SITES[1]).id)
    qpos[2] -= min(data.site_xpos[site, 2] for site in sole_sites)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)  # ty: ignore[unresolved-attribute]
    mujoco.mj_comPos(model, data)  # ty: ignore[unresolved-attribute]
    root_body = int(np.flatnonzero(model.body_parentid == 0)[1])
    limits = {joint.name: joint.limits for joint in load(variant).graph.joints}
    efforts = np.array(
        [
            limit.effort if (limit := limits.get(name)) is not None else np.inf
            for name in joint_names
        ]
    )
    speeds = np.array(
        [
            limit.velocity if (limit := limits.get(name)) is not None else np.inf
            for name in joint_names
        ]
    )
    solved = [
        *range(6),
        *(int(model.jnt_dofadr[model.joint(name).id]) for name in robot.leg_joints),
    ]
    return RobotModel(
        robot=robot,
        model=model,
        joint_names=joint_names,
        joint_position_addresses=np.array([model.jnt_qposadr[joint] for joint in hinges]),
        joint_velocity_addresses=np.array([model.jnt_dofadr[joint] for joint in hinges]),
        root_body=root_body,
        sole_sites=sole_sites,
        sole_half_extents=extents,
        pad_site=int(model.site(PAD_SITE).id),
        home_joint_positions=np.array([pose.get(name, 0.0) for name in joint_names]),
        home_configuration=qpos,
        stance_width=float(np.abs(np.diff(data.site_xpos[list(sole_sites), 1]))[0]),
        center_of_mass_height=float(data.subtree_com[root_body, 2]),
        height=_standing_height(model, data),
        mass=float(model.body_subtreemass[root_body]),
        lower_limits=np.array(
            [model.jnt_range[joint, 0] if model.jnt_limited[joint] else -np.inf for joint in hinges]
        ),
        upper_limits=np.array(
            [model.jnt_range[joint, 1] if model.jnt_limited[joint] else np.inf for joint in hinges]
        ),
        effort_limits=efforts,
        velocity_limits=speeds,
        solved_velocity_indices=np.array(solved),
        arm_swing=_arm_swing(model, robot, qpos),
    )


__all__ = ["PAD_SITE", "ROOT_JOINT", "SOLE_SITES", "RobotModel", "build_robot_model"]
