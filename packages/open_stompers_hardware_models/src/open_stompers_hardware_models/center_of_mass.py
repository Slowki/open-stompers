"""Robot center of mass from URDF link inertias and joint states."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

_ZERO_VECTOR: Final[NDArray[np.float64]] = np.zeros(3, dtype=np.float64)


@dataclass(frozen=True)
class _Joint:
    """URDF joint kinematics needed for center of mass calculation."""

    name: str
    parent_link: str
    child_link: str
    joint_type: str
    origin_xyz: NDArray[np.float64]
    origin_rpy: NDArray[np.float64]
    axis: NDArray[np.float64]


@dataclass(frozen=True)
class _LinkMass:
    """Link mass and its inertial center in link coordinates."""

    mass: float
    center: NDArray[np.float64]


class RobotCenterOfMass:
    """URDF-based center of mass calculator."""

    def __init__(self, urdf_path: Path) -> None:
        """Read link masses and joint kinematics from a URDF file."""
        robot = ET.parse(urdf_path).getroot()
        self.root_link = _root_link(robot)
        self.joints = tuple(_joint(element) for element in robot.findall("joint"))
        self.link_masses: dict[str, _LinkMass] = {}
        for element in robot.findall("link"):
            name = element.get("name")
            mass = _link_mass(element)
            if name is not None and mass is not None:
                self.link_masses[name] = mass
        if not self.link_masses:
            raise ValueError(f"URDF has no link inertial masses: {urdf_path}")
        self.movable_joint_names = {
            joint.name for joint in self.joints if joint.joint_type != "fixed"
        }

    def compute(
        self,
        joint_names: tuple[str, ...],
        joint_positions: NDArray[np.floating],
        root_position: NDArray[np.floating],
        root_quaternion_wxyz: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Compute the world-space center of mass for one sample or a trajectory.

        Args:
            joint_names: Names of the movable joints in `joint_positions` order.
            joint_positions: Joint values with shape `(..., joints)`.
            root_position: Root link positions with shape `(..., 3)`.
            root_quaternion_wxyz: Root link WXYZ quaternions with shape `(..., 4)`.

        Returns:
            Centers of mass with shape `(..., 3)`.
        """
        if set(joint_names) != self.movable_joint_names:
            raise ValueError(
                "Motion joints do not match movable URDF joints: "
                f"missing={sorted(self.movable_joint_names - set(joint_names))}, "
                f"unknown={sorted(set(joint_names) - self.movable_joint_names)}"
            )
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        batch_shape = joint_positions.shape[:-1]
        positions = joint_positions.reshape(-1, len(joint_names))
        joint_values = {name: positions[:, index] for index, name in enumerate(joint_names)}
        link_transforms = {
            self.root_link: (
                _quaternion_matrix(np.asarray(root_quaternion_wxyz).reshape(-1, 4)),
                np.asarray(root_position, dtype=np.float64).reshape(-1, 3),
            )
        }
        zeros = np.zeros(len(positions))
        unresolved = list(self.joints)
        while unresolved:
            remaining = []
            resolved_count = 0
            for joint in unresolved:
                if joint.parent_link not in link_transforms:
                    remaining.append(joint)
                    continue
                parent_rotation, parent_position = link_transforms[joint.parent_link]
                origin_rotation = _rpy_matrix(joint.origin_rpy)
                values = joint_values[joint.name] if joint.joint_type != "fixed" else zeros
                if joint.joint_type in {"revolute", "continuous"}:
                    motion_rotation = _axis_angle_matrix(joint.axis, values)
                    child_position = parent_position + parent_rotation @ joint.origin_xyz
                elif joint.joint_type == "prismatic":
                    motion_rotation = np.broadcast_to(np.eye(3), (len(values), 3, 3))
                    offsets = joint.origin_xyz + (origin_rotation @ joint.axis) * values[:, None]
                    child_position = parent_position + np.einsum(
                        "nij,nj->ni", parent_rotation, offsets
                    )
                elif joint.joint_type == "fixed":
                    motion_rotation = np.broadcast_to(np.eye(3), (len(values), 3, 3))
                    child_position = parent_position + parent_rotation @ joint.origin_xyz
                else:
                    raise ValueError(f"Unsupported URDF joint type: {joint.joint_type}")
                link_transforms[joint.child_link] = (
                    parent_rotation @ origin_rotation @ motion_rotation,
                    child_position,
                )
                resolved_count += 1
            if not resolved_count:
                raise ValueError("URDF joint graph does not connect every link to the root")
            unresolved = remaining

        weighted_center = np.zeros((len(positions), 3), dtype=np.float64)
        total_mass = 0.0
        for link_name, link_mass in self.link_masses.items():
            try:
                rotation, position = link_transforms[link_name]
            except KeyError as error:
                raise ValueError(f"Mass-bearing link is disconnected: {link_name}") from error
            weighted_center += link_mass.mass * (position + rotation @ link_mass.center)
            total_mass += link_mass.mass
        if total_mass <= 0:
            raise ValueError("URDF total mass must be positive")
        return (weighted_center / total_mass).reshape(*batch_shape, 3)


def _root_link(robot: ET.Element) -> str:
    """Find the root link that has no parent joint."""
    child_links = {
        child.get("link")
        for joint in robot.findall("joint")
        if (child := joint.find("child")) is not None
    }
    for link in robot.findall("link"):
        if (name := link.get("name")) is not None and name not in child_links:
            return name
    raise ValueError("URDF has no root link")


def _link_mass(link: ET.Element) -> _LinkMass | None:
    """Read a link mass and inertial origin."""
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_element = inertial.find("mass")
    if mass_element is None or (mass_text := mass_element.get("value")) is None:
        return None
    origin = inertial.find("origin")
    center = _vector(origin.get("xyz") if origin is not None else None, _ZERO_VECTOR)
    mass = float(mass_text)
    if not math.isfinite(mass) or mass < 0:
        raise ValueError(f"Invalid mass for link {link.get('name')}")
    return _LinkMass(mass, center)


def _joint(element: ET.Element) -> _Joint:
    """Read one URDF joint's parent, child, origin, and axis."""
    parent = element.find("parent")
    child = element.find("child")
    if parent is None or child is None:
        raise ValueError(f"Joint {element.get('name')} must specify parent and child links")
    origin = element.find("origin")
    axis_element = element.find("axis")
    axis = _vector(
        axis_element.get("xyz") if axis_element is not None else None, np.array([1, 0, 0])
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        raise ValueError(f"Joint {element.get('name')} has a zero axis")
    return _Joint(
        name=element.get("name", ""),
        parent_link=parent.get("link", ""),
        child_link=child.get("link", ""),
        joint_type=element.get("type", ""),
        origin_xyz=_vector(origin.get("xyz") if origin is not None else None, _ZERO_VECTOR),
        origin_rpy=_vector(origin.get("rpy") if origin is not None else None, _ZERO_VECTOR),
        axis=axis / axis_norm,
    )


def _vector(value: str | None, default: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parse a three-component URDF vector."""
    vector = (
        np.fromstring(value, sep=" ", dtype=np.float64) if value is not None else default.copy()
    )
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid three-component URDF vector: {value!r}")
    return vector


def _rpy_matrix(rpy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert URDF roll, pitch, yaw angles to a rotation matrix."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(
    axis: NDArray[np.float64], angles: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert rotations about one axis to matrices with shape `(len(angles), 3, 3)`."""
    x, y, z = axis
    cosine = np.cos(angles)
    sine = np.sin(angles)
    complement = 1 - cosine
    return np.stack(
        (
            np.stack(
                (
                    cosine + x * x * complement,
                    x * y * complement - z * sine,
                    x * z * complement + y * sine,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    y * x * complement + z * sine,
                    cosine + y * y * complement,
                    y * z * complement - x * sine,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    z * x * complement - y * sine,
                    z * y * complement + x * sine,
                    cosine + z * z * complement,
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )


def _quaternion_matrix(quaternions_wxyz: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert WXYZ quaternions with shape `(n, 4)` to rotation matrices."""
    w, x, y, z = np.asarray(quaternions_wxyz, dtype=np.float64).T
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )
