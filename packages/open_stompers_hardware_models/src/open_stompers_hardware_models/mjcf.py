"""Export LinkForge robot models to MuJoCo XML."""

import xml.etree.ElementTree as ET
from pathlib import Path

from linkforge.core import (
    Box,
    Collision,
    Cylinder,
    Geometry,
    Joint,
    JointType,
    Mesh,
    Robot,
    Sensor,
    SensorType,
    Sphere,
    Visual,
)

from open_stompers_hardware_models import assets
from open_stompers_hardware_models.mjcf_metadata import MjcfMetadata, MjcfSensor, MjcfSite
from open_stompers_hardware_models.robots.microduck.transforms import (
    BODY_QUATERNIONS,
    GEOM_QUATERNIONS,
)
from open_stompers_hardware_models.sensors import camera_parameters


class MjcfExportError(ValueError):
    """Raised when a robot cannot be represented as MJCF."""


def _numbers(*values: float) -> str:
    """Format numeric MJCF attributes."""
    return " ".join(f"{value:.12g}" for value in values)


def _geometry_attributes(geometry: Geometry, mesh_names: dict[Mesh, str]) -> dict[str, str]:
    """Build MJCF attributes for geometry."""
    match geometry:
        case Box(size=size):
            return {"type": "box", "size": _numbers(size.x / 2, size.y / 2, size.z / 2)}
        case Cylinder(radius=radius, length=length):
            return {"type": "cylinder", "size": _numbers(radius, length / 2)}
        case Sphere(radius=radius):
            return {"type": "sphere", "size": _numbers(radius)}
        case Mesh():
            return {"type": "mesh", "mesh": mesh_names[geometry]}
        case _:
            raise MjcfExportError(f"Unsupported geometry: {type(geometry).__name__}")


def _add_geometry(
    body_element: ET.Element,
    robot: Robot,
    geometry_item: Visual | Collision,
    mesh_names: dict[Mesh, str],
    *,
    collision: bool,
    occurrence: int,
) -> None:
    """Add one geometry element to a body."""
    geometry = geometry_item.geometry
    attributes = _geometry_attributes(geometry, mesh_names)
    attributes["pos"] = _numbers(*geometry_item.origin.xyz)
    mesh_name = (
        geometry.resource.rsplit("/", maxsplit=1)[-1].removesuffix(".stl")
        if isinstance(geometry, Mesh)
        else None
    )
    upstream_quaternion = (
        GEOM_QUATERNIONS.get((body_element.attrib["name"], mesh_name, collision, occurrence))
        if mesh_name is not None
        else None
    )
    if upstream_quaternion is None:
        attributes["euler"] = _numbers(*geometry_item.origin.rpy)
    else:
        attributes["quat"] = _numbers(*upstream_quaternion)
    if geometry_item.name:
        attributes["name"] = geometry_item.name
    if collision:
        attributes["friction"] = _numbers(
            robot.link(body_element.attrib["name"]).physics.mu,
            robot.link(body_element.attrib["name"]).physics.mu2,
            0.0001,
        )
    else:
        attributes.update({"contype": "0", "conaffinity": "0", "group": "2"})
        if (
            isinstance(geometry_item, Visual)
            and geometry_item.material
            and geometry_item.material.color
        ):
            attributes["rgba"] = _numbers(*geometry_item.material.color.to_tuple())
    ET.SubElement(body_element, "geom", attributes)


def _add_joint(body_element: ET.Element, joint: Joint) -> None:
    """Add a joint element to a body."""
    match joint.type:
        case JointType.FIXED:
            return
        case JointType.REVOLUTE | JointType.CONTINUOUS:
            if joint.axis is None:
                raise MjcfExportError(f"Joint '{joint.name}' is missing an axis")
            attributes = {"name": joint.name, "type": "hinge", "axis": _numbers(*joint.axis)}
        case JointType.PRISMATIC:
            if joint.axis is None:
                raise MjcfExportError(f"Joint '{joint.name}' is missing an axis")
            attributes = {"name": joint.name, "type": "slide", "axis": _numbers(*joint.axis)}
        case JointType.FLOATING:
            attributes = {"name": joint.name, "type": "free"}
        case _:
            raise MjcfExportError(f"Joint '{joint.name}' has unsupported type '{joint.type}'")
    if joint.limits and joint.limits.lower is not None and joint.limits.upper is not None:
        attributes.update(
            {"limited": "true", "range": _numbers(joint.limits.lower, joint.limits.upper)}
        )
    if joint.dynamics:
        attributes.update(
            {
                "damping": _numbers(joint.dynamics.damping),
                "frictionloss": _numbers(joint.dynamics.friction),
            }
        )
    ET.SubElement(body_element, "joint", attributes)


def _add_inertial(body_element: ET.Element, robot: Robot, link_name: str) -> None:
    """Add explicit link mass and inertia."""
    if (link := robot.link(link_name)).inertial is None:
        return
    inertial = link.inertial
    tensor = inertial.inertia
    ET.SubElement(
        body_element,
        "inertial",
        {
            "pos": _numbers(*inertial.origin.xyz),
            "mass": _numbers(inertial.mass),
            "fullinertia": _numbers(
                tensor.ixx,
                tensor.iyy,
                tensor.izz,
                tensor.ixy,
                tensor.ixz,
                tensor.iyz,
            ),
        },
    )


def _mesh_path(robot: Robot, mesh: Mesh) -> Path:
    """Resolve a mesh resource."""
    try:
        return assets.resolve(mesh.resource)
    except assets.AssetNotFoundError:
        return robot.resolve_resource(mesh.resource)


class _MjcfBuilder:
    """Build one MJCF document while retaining export-local state."""

    def __init__(self, robot: Robot, metadata: MjcfMetadata) -> None:
        self.robot = robot
        self.metadata = metadata
        self.root = ET.Element("mujoco", {"model": robot.name})
        ET.SubElement(self.root, "compiler", {"angle": "radian", "eulerseq": "XYZ"})
        self.mesh_names: dict[Mesh, str] = {}
        self.geometry_occurrences: dict[tuple[str, bool, str], int] = {}

    def _add_assets(self) -> None:
        asset_element = ET.SubElement(self.root, "asset")
        meshes = {
            item.geometry
            for link in self.robot.links
            for item in (*link.visuals, *link.collisions)
            if isinstance(item.geometry, Mesh)
        }
        for index, mesh in enumerate(sorted(meshes, key=lambda item: item.resource)):
            mesh_name = f"mesh_{index}"
            self.mesh_names[mesh] = mesh_name
            ET.SubElement(
                asset_element,
                "mesh",
                {
                    "name": mesh_name,
                    "file": str(_mesh_path(self.robot, mesh).resolve()),
                    "scale": _numbers(*mesh.scale),
                },
            )

    @staticmethod
    def _body_attributes(link_name: str, joint: Joint | None = None) -> dict[str, str]:
        """Build a body transform from the upstream Microduck frame when available."""
        attributes = {"name": link_name}
        if joint is not None:
            attributes["pos"] = _numbers(*joint.origin.xyz)
            upstream_quaternion = BODY_QUATERNIONS.get(link_name)
            if upstream_quaternion is None:
                attributes["euler"] = _numbers(*joint.origin.rpy)
            else:
                attributes["quat"] = _numbers(*upstream_quaternion)
        return attributes

    def _add_link_geometry(self, body_element: ET.Element, link_name: str) -> None:
        """Add all link geometry with occurrence-aware upstream transforms."""
        link = self.robot.link(link_name)
        for collision, items in ((True, link.collisions), (False, link.visuals)):
            for item in items:
                geometry = item.geometry
                mesh_name = (
                    geometry.resource.rsplit("/", maxsplit=1)[-1]
                    if isinstance(geometry, Mesh)
                    else type(geometry).__name__
                )
                key = (link_name, collision, mesh_name)
                occurrence = self.geometry_occurrences.get(key, 0)
                self.geometry_occurrences[key] = occurrence + 1
                _add_geometry(
                    body_element,
                    self.robot,
                    item,
                    self.mesh_names,
                    collision=collision,
                    occurrence=occurrence,
                )

    def _add_body(
        self, link_name: str, parent_element: ET.Element, joint: Joint | None = None
    ) -> None:
        """Add a link body and its descendants."""
        body_element = ET.SubElement(
            parent_element, "body", self._body_attributes(link_name, joint)
        )
        if joint is not None:
            _add_joint(body_element, joint)
        _add_inertial(body_element, self.robot, link_name)
        self._add_link_geometry(body_element, link_name)
        for child_joint in self.robot.get_child_joints(link_name):
            self._add_body(child_joint.child, body_element, child_joint)

    def _add_body_tree(self) -> dict[str, ET.Element]:
        world_element = ET.SubElement(self.root, "worldbody")
        root_link = self.robot.root_link.name
        root_attributes = self._body_attributes(root_link)
        if upstream_quaternion := BODY_QUATERNIONS.get(root_link):
            root_attributes["quat"] = _numbers(*upstream_quaternion)
        if self.metadata.root_pos is not None:
            root_attributes["pos"] = _numbers(*self.metadata.root_pos)
        root_body = ET.SubElement(world_element, "body", root_attributes)
        if self.metadata.root_freejoint is not None:
            ET.SubElement(root_body, "freejoint", {"name": self.metadata.root_freejoint})
        _add_inertial(root_body, self.robot, root_link)
        self._add_link_geometry(root_body, root_link)
        for child_joint in self.robot.get_child_joints(root_link):
            self._add_body(child_joint.child, root_body, child_joint)
        return {body.attrib["name"]: body for body in self.root.iter("body")}

    def _add_sites(self, bodies: dict[str, ET.Element]) -> tuple[tuple[MjcfSite, ...], set[str]]:
        sites: tuple[MjcfSite, ...] = self.metadata.sites
        for site in sites:
            if site.body not in bodies:
                raise MjcfExportError(f"Site {site.name!r} references missing body {site.body!r}")
            ET.SubElement(
                bodies[site.body],
                "site",
                {
                    "name": site.name,
                    "pos": _numbers(*site.pos),
                    "quat": _numbers(*site.quat),
                    "group": "3",
                },
            )
        return sites, {site.name for site in sites}

    @staticmethod
    def _camera(body: ET.Element, sensor: Sensor) -> None:
        camera = sensor.camera_info
        if camera is None:
            raise MjcfExportError(f"Camera sensor '{sensor.name}' is missing camera info")
        axes, vertical_fov = camera_parameters(sensor)
        ET.SubElement(
            body,
            "camera",
            name=sensor.name,
            pos=_numbers(*sensor.origin.xyz),
            xyaxes=_numbers(*axes),
            fovy=_numbers(vertical_fov),
            resolution=f"{camera.width} {camera.height}",
            output="depth" if sensor.type is SensorType.DEPTH_CAMERA else "rgb",
        )

    @staticmethod
    def _imu_sensors(sensor: Sensor, configured: tuple[MjcfSensor, ...]) -> list[MjcfSensor]:
        information = sensor.imu_info
        if information is None:
            raise MjcfExportError(f"IMU sensor '{sensor.name}' is missing IMU info")
        generated = []
        for suffix, kind, noise in (
            ("orientation", "framequat", None),
            ("ang_vel", "gyro", information.angular_velocity_noise),
            ("accel", "accelerometer", information.linear_acceleration_noise),
        ):
            name = f"{sensor.name}_{suffix}"
            if not any(existing.name == name for existing in configured):
                generated.append(
                    MjcfSensor(name, kind, sensor.name, noise=noise.stddev if noise else 0.0)
                )
        return generated

    def _add_robot_sensors(
        self,
        bodies: dict[str, ET.Element],
        site_names: set[str],
        configured: tuple[MjcfSensor, ...],
    ) -> tuple[MjcfSensor, ...]:
        generated: list[MjcfSensor] = []
        for sensor in self.robot.sensors:
            if sensor.link_name not in bodies:
                raise MjcfExportError(f"Sensor {sensor.name!r} references a missing body")
            if sensor.name in site_names:
                raise MjcfExportError(f"Duplicate sensor site {sensor.name!r}")
            site_names.add(sensor.name)
            body = bodies[sensor.link_name]
            ET.SubElement(
                body,
                "site",
                name=sensor.name,
                pos=_numbers(*sensor.origin.xyz),
                euler=_numbers(*sensor.origin.rpy),
                group="3",
            )
            match sensor.type:
                case SensorType.CAMERA | SensorType.DEPTH_CAMERA:
                    self._camera(body, sensor)
                case SensorType.IMU:
                    generated.extend(self._imu_sensors(sensor, configured))
                case _:
                    raise MjcfExportError(f"Unsupported sensor type: {sensor.type.value}")
        return (*configured, *generated)

    def _add_sensor_elements(
        self,
        sensors: tuple[MjcfSensor, ...],
        bodies: dict[str, ET.Element],
        site_names: set[str],
    ) -> None:
        if not sensors:
            return
        sensor_element = ET.SubElement(self.root, "sensor")
        for sensor in sensors:
            attributes = {"name": sensor.name, "noise": _numbers(sensor.noise)}
            if sensor.kind == "subtreeangmom":
                if sensor.target not in bodies:
                    raise MjcfExportError(f"Sensor {sensor.name!r} references a missing body")
                attributes["body"] = sensor.target
            else:
                if sensor.target not in site_names:
                    raise MjcfExportError(f"Sensor {sensor.name!r} references a missing site")
                if sensor.kind == "framequat":
                    attributes.update(objtype="site", objname=sensor.target)
                else:
                    attributes["site"] = sensor.target
            ET.SubElement(sensor_element, sensor.kind, attributes)

    def write(self, output: str | Path) -> Path:
        """Build and write the complete document."""
        self._add_assets()
        bodies = self._add_body_tree()
        _, site_names = self._add_sites(bodies)
        configured: tuple[MjcfSensor, ...] = self.metadata.sensors
        sensors = self._add_robot_sensors(bodies, site_names, configured)
        self._add_sensor_elements(sensors, bodies, site_names)
        ET.indent(self.root)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.root).write(destination, encoding="unicode", xml_declaration=True)
        return destination


def export_mjcf(
    robot: Robot,
    output: str | Path,
    *,
    mjcf_metadata: MjcfMetadata | None = None,
) -> Path:
    """Export a robot as MuJoCo XML.

    Args:
        robot: Validated LinkForge robot to export.
        output: Destination path for the MJCF file.
        mjcf_metadata: Optional MuJoCo-specific data for the robot.

    Returns:
        The written output path.
    """
    metadata = mjcf_metadata if mjcf_metadata is not None else MjcfMetadata()
    return _MjcfBuilder(robot, metadata).write(output)


__all__ = ["MjcfExportError", "export_mjcf"]
