"""Shared LinkForge building blocks for robot hardware models.

Robot models are described as declarative link and joint specs and compiled into
LinkForge ``Robot`` models via the Composer API.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from linkforge.core import (
    CameraInfo,
    Geometry,
    InertiaTensor,
    JointLimits,
    JointType,
    LinkBuilder,
    Robot,
    RobotBuilder,
    Sensor,
    SensorType,
    Transform,
    Vector3,
    box,
    cylinder,
    mesh,
    sphere,
    validate_robot,
)

JointKind = Literal["revolute", "continuous", "fixed"]
CollisionKind = Literal["box", "cylinder", "sphere", "mesh"]


class ModelDefinitionError(RuntimeError):
    """Raised when a model spec table is inconsistent or fails validation."""


@dataclass(frozen=True, slots=True)
class CollisionSpec:
    """A collision geometry attached to a link.

    Attributes:
        kind: Primitive or mesh type.
        dims: Box sizes (x, y, z), cylinder (radius, length), or sphere (radius,).
            Unused for mesh collisions.
        mesh: Mesh file name for ``kind="mesh"``, resolved via the model's mesh
            resource function.
        xyz: Translation of the collision frame relative to the link frame.
        rpy: Rotation of the collision frame relative to the link frame.
    """

    kind: CollisionKind
    dims: tuple[float, ...] = ()
    mesh: str | None = None
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class VisualSpec:
    """A mesh visual attached to a link.

    Attributes:
        mesh: Mesh file name, resolved via the model's mesh resource function.
        xyz: Translation of the visual frame relative to the link frame.
        rpy: Rotation of the visual frame relative to the link frame.
        material: Material name. Falls back to the link or robot default.
    """

    mesh: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str | None = None


@dataclass(frozen=True, slots=True)
class ImuSpec:
    """An IMU attached to a link.

    Attributes:
        name: Sensor name.
        xyz: Translation relative to the link frame.
        rpy: Rotation relative to the link frame.
        update_rate: Sampling rate in hertz. ``None`` uses the LinkForge default.
    """

    name: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    update_rate: float | None = None


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """A camera or depth camera attached to a link.

    Attributes:
        name: Sensor name.
        xyz: Translation relative to the link frame.
        rpy: Rotation relative to the link frame.
        fov: Horizontal field of view in radians. ``None`` uses the LinkForge default
            for RGB cameras and is required for depth cameras.
        width: Image width in pixels. ``None`` uses the LinkForge default for RGB
            cameras and is required for depth cameras.
        height: Image height in pixels. ``None`` uses the LinkForge default for RGB
            cameras and is required for depth cameras.
        depth: Whether this is a depth camera rather than an RGB camera.
    """

    name: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float | None = None
    width: int | None = None
    height: int | None = None
    depth: bool = False


@dataclass(frozen=True, slots=True)
class LinkSpec:
    """A rigid body with visuals, inertials, collisions, and sensors.

    Attributes:
        name: Link name.
        mass: Mass in kilograms.
        com: Center of mass position in the link frame.
        inertia: Inertia tensor (ixx, iyy, izz, ixy, ixz, iyz) about the COM.
        mesh: Single visual mesh at the link origin. Mutually exclusive with
            ``visuals``.
        visuals: Mesh visuals with their own frames and materials.
        collisions: Collision geometries.
        imus: IMUs attached to this link.
        cameras: Cameras attached to this link.
    """

    name: str
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]
    mesh: str | None = None
    visuals: tuple[VisualSpec, ...] = ()
    collisions: tuple[CollisionSpec, ...] = ()
    imus: tuple[ImuSpec, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class JointSpec:
    """A joint connecting ``child`` to ``parent``.

    Attributes:
        name: Joint name.
        parent: Parent link name.
        child: Child link name.
        kind: Joint type.
        xyz: Joint origin translation in the parent frame.
        rpy: Joint origin rotation in the parent frame.
        axis: Joint axis (revolute/continuous only).
        limits: (lower, upper) position limits in radians (revolute only).
        effort: Effort limit.
        velocity: Velocity limit.
        damping: Joint damping.
        friction: Joint friction.
    """

    name: str
    parent: str
    child: str
    kind: JointKind
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limits: tuple[float, float] | None = None
    effort: float = 0.0
    velocity: float = 0.0
    damping: float = 0.0
    friction: float = 0.0


def _collision_geometry(spec: CollisionSpec, mesh_resource: Callable[[str], str]) -> Geometry:
    match spec.kind:
        case "box":
            if len(spec.dims) != 3:
                raise ModelDefinitionError(f"Box collision needs 3 dimensions, got {spec.dims}")
            return box(*spec.dims)
        case "cylinder":
            if len(spec.dims) != 2:
                raise ModelDefinitionError(
                    f"Cylinder collision needs (radius, length), got {spec.dims}"
                )
            return cylinder(radius=spec.dims[0], length=spec.dims[1])
        case "sphere":
            if len(spec.dims) != 1:
                raise ModelDefinitionError(f"Sphere collision needs (radius,), got {spec.dims}")
            return sphere(radius=spec.dims[0])
        case "mesh":
            if spec.mesh is None:
                raise ModelDefinitionError("Mesh collision is missing a mesh file name")
            return mesh(mesh_resource(spec.mesh))


def _visual_specs(
    spec: LinkSpec,
    material_by_link: Mapping[str, str],
    default_material: str | None,
) -> tuple[VisualSpec, ...]:
    if spec.mesh is not None and spec.visuals:
        raise ModelDefinitionError(f"Link '{spec.name}' sets both mesh and visuals")
    if spec.mesh is not None:
        return (
            VisualSpec(
                mesh=spec.mesh,
                material=material_by_link.get(spec.name, default_material),
            ),
        )
    fallback = material_by_link.get(spec.name, default_material)
    return tuple(
        visual
        if visual.material is not None or fallback is None
        else replace(visual, material=fallback)
        for visual in spec.visuals
    )


def _add_sensors(link_builder: LinkBuilder, spec: LinkSpec) -> None:
    for imu in spec.imus:
        if imu.update_rate is None:
            link_builder.imu(name=imu.name, xyz=imu.xyz, rpy=imu.rpy)
        else:
            link_builder.imu(
                name=imu.name,
                update_rate=imu.update_rate,
                xyz=imu.xyz,
                rpy=imu.rpy,
            )

    for camera in spec.cameras:
        if camera.depth:
            if camera.fov is None or camera.width is None or camera.height is None:
                raise ModelDefinitionError(
                    f"Depth camera '{camera.name}' on '{spec.name}' needs fov, width, and height"
                )
            link_builder.sensor(
                Sensor(
                    name=camera.name,
                    type=SensorType.DEPTH_CAMERA,
                    link_name=spec.name,
                    camera_info=CameraInfo(
                        horizontal_fov=camera.fov,
                        width=camera.width,
                        height=camera.height,
                    ),
                    origin=Transform(xyz=Vector3(*camera.xyz), rpy=Vector3(*camera.rpy)),
                )
            )
            continue
        if camera.fov is None and camera.width is None and camera.height is None:
            link_builder.camera(name=camera.name, xyz=camera.xyz, rpy=camera.rpy)
            continue
        if camera.fov is None or camera.width is None or camera.height is None:
            raise ModelDefinitionError(
                f"Camera '{camera.name}' on '{spec.name}' must set fov, width, and height together"
            )
        link_builder.camera(
            name=camera.name,
            fov=camera.fov,
            width=camera.width,
            height=camera.height,
            xyz=camera.xyz,
            rpy=camera.rpy,
        )


def _add_link(
    builder: RobotBuilder,
    spec: LinkSpec,
    joint: JointSpec | None,
    mesh_resource: Callable[[str], str],
    material_by_link: Mapping[str, str],
    default_material: str | None,
) -> None:
    if joint is None:
        link_builder = builder.link(spec.name)
    else:
        link_builder = builder.link(spec.name, parent=joint.parent, joint_name=joint.name)

    for visual in _visual_specs(spec, material_by_link, default_material):
        link_builder.visual(
            mesh(mesh_resource(visual.mesh)),
            xyz=visual.xyz,
            rpy=visual.rpy,
            material=visual.material,
        )
    for collision in spec.collisions:
        link_builder.collision(
            _collision_geometry(collision, mesh_resource),
            xyz=collision.xyz,
            rpy=collision.rpy,
        )
    _add_sensors(link_builder, spec)

    ixx, iyy, izz, ixy, ixz, iyz = spec.inertia
    link_builder.mass(
        spec.mass,
        origin_xyz=spec.com,
        inertia=InertiaTensor(ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz),
    )

    if joint is None:
        link_builder.root()
        return

    match joint.kind:
        case "fixed":
            link_builder.fixed(xyz=joint.xyz, rpy=joint.rpy)
        case "revolute":
            if joint.limits is None:
                raise ValueError(f"revolute joint '{joint.name}' has no limits")
            link_builder.revolute(
                axis=joint.axis,
                limits=joint.limits,
                xyz=joint.xyz,
                rpy=joint.rpy,
                effort=joint.effort,
                velocity=joint.velocity,
            )
            _apply_dynamics(link_builder, joint)
        case "continuous":
            # LinkForge stores effort on a limit whose lower/upper default to 0.
            # Replace that so a continuous joint is not a locked hinge.
            link_builder.continuous(axis=joint.axis, xyz=joint.xyz, rpy=joint.rpy)
            link_builder._joint.limits = (
                JointLimits(effort=joint.effort, velocity=joint.velocity)
                if joint.effort or joint.velocity
                else None
            )
            _apply_dynamics(link_builder, joint)
    link_builder.commit()


def _apply_dynamics(link_builder: LinkBuilder, joint: JointSpec) -> None:
    if joint.damping or joint.friction:
        link_builder.dynamics(damping=joint.damping, friction=joint.friction)


def build_robot(
    name: str,
    links: Sequence[LinkSpec],
    joints: Sequence[JointSpec],
    mesh_resource: Callable[[str], str],
    *,
    materials: Mapping[str, tuple[float, float, float, float]],
    default_material: str | None = None,
    material_by_link: Mapping[str, str] | None = None,
) -> Robot:
    """Compile spec tables into a validated LinkForge robot model.

    Args:
        name: Robot name.
        links: Link specs; the single link without an incoming joint is the root.
        joints: Joint specs; spec order must place parents before children.
        mesh_resource: Maps a mesh file name to its resource string.
        materials: RGBA colors registered on the robot, keyed by material name.
        default_material: Material used when a visual does not name one.
        material_by_link: Optional visual material override by link name. Applies
            to single-mesh links and to visuals that do not name a material.

    Returns:
        The validated LinkForge Robot.

    Raises:
        ModelDefinitionError: If the spec tables are inconsistent or the
            resulting robot fails LinkForge validation.
    """
    builder = RobotBuilder(name)
    for material_name, color in materials.items():
        builder.material(material_name, color=color)
    if default_material is not None and default_material not in materials:
        raise ModelDefinitionError(
            f"Robot '{name}': default material '{default_material}' is not registered"
        )

    material_by_link = material_by_link or {}
    link_names = {spec.name for spec in links}
    if unknown_material_links := set(material_by_link) - link_names:
        names = ", ".join(sorted(unknown_material_links))
        raise ModelDefinitionError(
            f"Robot '{name}': material overrides reference unknown links: {names}"
        )

    joint_by_child = {joint.child: joint for joint in joints}
    if len(joint_by_child) != len(joints):
        msg = f"Robot '{name}': duplicate child links in joint specs"
        raise ModelDefinitionError(msg)

    roots = [spec.name for spec in links if spec.name not in joint_by_child]
    if len(roots) != 1:
        msg = f"Robot '{name}': expected exactly one root link, got {roots}"
        raise ModelDefinitionError(msg)

    for spec in links:
        _add_link(
            builder,
            spec,
            joint_by_child.get(spec.name),
            mesh_resource,
            material_by_link,
            default_material,
        )

    robot = builder.build(validate=True)
    result = validate_robot(robot)
    if not result.is_valid:
        issues = "; ".join(str(issue) for issue in result.errors)
        msg = f"Robot '{name}' failed LinkForge validation: {issues}"
        raise ModelDefinitionError(msg)
    return robot


def actuated_joint_names(robot: Robot) -> list[str]:
    """Return the names of all non-fixed joints in graph order."""
    return [joint.name for joint in robot.graph.joints if joint.type is not JointType.FIXED]
