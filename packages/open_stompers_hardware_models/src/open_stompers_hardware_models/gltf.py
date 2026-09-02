"""Export LinkForge robot models to binary GLTF files."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import trimesh
from linkforge.core import Box, Cylinder, Geometry, Joint, JointType, Material, Mesh, Robot, Sphere
from trimesh.visual.material import PBRMaterial

from open_stompers_hardware_models import VARIANTS, assets, load
from open_stompers_hardware_models.materials import DEFAULT_PBR_MATERIAL, PbrMaterialSpec
from open_stompers_hardware_models.robots.microduck.materials import (
    PBR_MATERIALS as MICRODUCK_PBR_MATERIALS,
)
from open_stompers_hardware_models.robots.mini_pi_plus.materials import (
    PBR_MATERIALS as MINI_PI_PLUS_PBR_MATERIALS,
)
from open_stompers_hardware_models.variants import HardwareVariant


class GltfExportError(ValueError):
    """Raised when a robot pose cannot be exported."""


PBR_MATERIALS: Final[dict[str, PbrMaterialSpec]] = {
    **MICRODUCK_PBR_MATERIALS,
    **MINI_PI_PLUS_PBR_MATERIALS,
}


def _transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    """Build a homogeneous matrix from a LinkForge XYZ/RPY transform."""
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _axis_motion(joint: Joint, position: float) -> np.ndarray:
    """Return a joint motion matrix for one scalar joint position."""
    result = np.eye(4)
    if joint.axis is None:
        return result
    if joint.type is JointType.PRISMATIC:
        result[:3, 3] = np.asarray(tuple(joint.axis)) * position
        return result
    if joint.type not in (JointType.REVOLUTE, JointType.CONTINUOUS):
        return result

    axis = np.asarray(tuple(joint.axis), dtype=float)
    x, y, z = axis
    cosine, sine = math.cos(position), math.sin(position)
    one_minus_cosine = 1.0 - cosine
    result[:3, :3] = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )
    return result


def _geometry(robot: Robot, geometry: Geometry) -> trimesh.Trimesh:
    """Load or construct one visual geometry."""
    if isinstance(geometry, Mesh):
        try:
            resource = assets.resolve(geometry.resource)
        except assets.AssetNotFoundError:
            resource = robot.resolve_resource(geometry.resource)
        loaded = trimesh.load_mesh(resource, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.dump()))
        loaded.apply_scale(tuple(geometry.scale))
        return cast(trimesh.Trimesh, loaded)
    if isinstance(geometry, Box):
        return trimesh.creation.box(extents=tuple(geometry.size))
    if isinstance(geometry, Cylinder):
        return trimesh.creation.cylinder(radius=geometry.radius, height=geometry.length)
    if isinstance(geometry, Sphere):
        return trimesh.creation.uv_sphere(radius=geometry.radius)
    raise GltfExportError(f"Unsupported visual geometry: {type(geometry).__name__}")


def _simplify_mesh(mesh: trimesh.Trimesh, keep_ratio: float) -> trimesh.Trimesh:
    """Retain a fraction of a mesh's faces using quadric decimation."""
    if keep_ratio == 1.0 or len(mesh.faces) < 4:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(percent=1.0 - keep_ratio)
    except Exception as error:
        raise GltfExportError(f"Could not simplify mesh: {error}") from error


def _apply_material(mesh: trimesh.Trimesh, material: Material | None) -> None:
    """Apply a LinkForge material and PBR factors to a trimesh mesh."""
    if material is None:
        return

    if pbr_material := PBR_MATERIALS.get(material.name):
        base_color = pbr_material.base_color
        metallic = pbr_material.metallic
        roughness = pbr_material.roughness
    elif material.color is not None:
        base_color = material.color.to_tuple()
        metallic = 0.0
        roughness = 0.5
    else:
        base_color = DEFAULT_PBR_MATERIAL.base_color
        metallic = DEFAULT_PBR_MATERIAL.metallic
        roughness = DEFAULT_PBR_MATERIAL.roughness

    cast(Any, mesh.visual).material = PBRMaterial(
        name=material.name,
        baseColorFactor=base_color,
        metallicFactor=metallic,
        roughnessFactor=roughness,
    )


def _build_scene(
    robot: Robot,
    pose: Mapping[str, float] | None = None,
    *,
    simplify_ratio: float | None = None,
    stable_node_names: bool = False,
) -> trimesh.Scene:
    """Build a world-space GLTF scene from a robot pose."""
    if simplify_ratio is not None and (
        not math.isfinite(simplify_ratio) or not 0.0 < simplify_ratio <= 1.0
    ):
        raise GltfExportError("Simplification ratio must be finite and in (0, 1]")
    pose = pose or {}

    if unknown := set(pose) - {joint.name for joint in robot.graph.joints}:
        names = ", ".join(sorted(unknown))
        raise GltfExportError(f"Pose contains unknown joint(s): {names}")
    for joint in robot.graph.joints:
        position = pose.get(joint.name, 0.0)
        if not math.isfinite(position):
            raise GltfExportError(f"Pose for joint '{joint.name}' must be finite")
        if joint.type is JointType.FIXED and position != 0.0:
            raise GltfExportError(f"Fixed joint '{joint.name}' cannot have a nonzero pose")

    world_transforms: dict[str, np.ndarray] = {robot.root_link.name: np.eye(4)}
    for joint in robot.graph.get_topological_joints():
        parent = world_transforms[joint.parent]
        origin = _transform(
            (joint.origin.xyz.x, joint.origin.xyz.y, joint.origin.xyz.z),
            (joint.origin.rpy.x, joint.origin.rpy.y, joint.origin.rpy.z),
        )
        world_transforms[joint.child] = (
            parent @ origin @ _axis_motion(joint, pose.get(joint.name, 0.0))
        )

    scene = trimesh.Scene()
    # LinkForge/URDF uses +Z as up, while GLTF defines +Y as up. Convert the
    # complete robot after applying its native kinematics and visual poses.
    z_up_to_gltf = _transform((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0))
    for link_name in robot.graph.get_topological_link_names():
        link = robot.link(link_name)
        for index, visual in enumerate(link.visuals):
            mesh = _geometry(robot, visual.geometry)
            if simplify_ratio is not None and isinstance(visual.geometry, Mesh):
                mesh = _simplify_mesh(mesh, simplify_ratio)
            _apply_material(mesh, visual.material)
            visual_transform = _transform(
                (visual.origin.xyz.x, visual.origin.xyz.y, visual.origin.xyz.z),
                (visual.origin.rpy.x, visual.origin.rpy.y, visual.origin.rpy.z),
            )
            mesh.apply_transform(z_up_to_gltf @ world_transforms[link_name] @ visual_transform)

            scene.add_geometry(
                mesh,
                node_name=(
                    f"open_stompers_visual_{len(scene.geometry):04d}"
                    if stable_node_names
                    else visual.name or f"{link_name}_visual_{index}"
                ),
            )

    if not scene.geometry:
        raise GltfExportError(f"Robot '{robot.name}' has no visual geometry")
    return scene


def export_glb(
    robot: Robot,
    output: str | Path,
    pose: Mapping[str, float] | None = None,
    *,
    simplify_ratio: float | None = None,
) -> Path:
    """Export a robot's visual model as a binary GLB file.

    Args:
        robot: Validated LinkForge robot to export.
        output: Destination path for the GLB file.
        pose: Optional joint-position mapping. Revolute values are radians and
            prismatic values are meters; omitted joints default to zero.
        simplify_ratio: Optional fraction of STL mesh faces to retain. Values
            must be in ``(0, 1]``; omitted values leave meshes unchanged.

    Returns:
        The written output path.

    Raises:
        GltfExportError: If the pose or simplification ratio is invalid, or
            mesh decimation fails.
    """
    scene = _build_scene(robot, pose, simplify_ratio=simplify_ratio)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination, file_type="glb")
    return destination


def export_all(
    output_dir: str | Path,
    poses: Mapping[HardwareVariant, Mapping[str, float]] | None = None,
    variants: tuple[HardwareVariant, ...] | None = None,
    *,
    simplify_ratio: float | None = None,
) -> list[Path]:
    """Export one zero-pose GLB per named robot.

    Args:
        output_dir: Directory receiving ``<name>.glb`` files.
        poses: Optional per-robot joint pose mappings.
        variants: Optional variants to export; defaults to every known variant.
        simplify_ratio: Optional fraction of STL mesh faces to retain.

    Returns:
        Paths to the generated GLB files in mapping order.
    """
    names = variants or VARIANTS
    robots = {name: load(name) for name in names}
    poses = poses or {}

    if unknown := set(poses) - set(robots):
        raise GltfExportError(f"Poses supplied for unknown robot(s): {', '.join(sorted(unknown))}")
    directory = Path(output_dir)
    return [
        export_glb(
            robot,
            directory / f"{name}.glb",
            poses.get(name),
            simplify_ratio=simplify_ratio,
        )
        for name, robot in robots.items()
    ]


__all__ = ["GltfExportError", "export_all", "export_glb"]
