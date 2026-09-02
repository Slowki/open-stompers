"""Hardware model export dispatch."""

import xml.etree.ElementTree as ET
from enum import StrEnum
from os import path as operating_system_path
from pathlib import Path
from typing import Final

from linkforge.core import Robot, URDFGenerator

from open_stompers_hardware_models import assets
from open_stompers_hardware_models.blender_export import (
    BlenderExportError,
    _quaternion_rpy,
    _source_body_rpy,
    export_blender,
)
from open_stompers_hardware_models.gltf import GltfExportError, export_glb
from open_stompers_hardware_models.mjcf import export_mjcf
from open_stompers_hardware_models.mjcf_metadata import MjcfMetadata
from open_stompers_hardware_models.robots.microduck.transforms import GEOM_QUATERNIONS


class ExportFormat(StrEnum):
    """Supported hardware model export format."""

    BLENDER = "blender"
    GLB = "glb"
    URDF = "urdf"
    MJCF = "mjcf"


_MESH_LINK: Final[Path] = Path(__file__).resolve().parents[2] / "data"
_CACHE_RESOURCE_PREFIX: Final[str] = "hardware_models/"


def _ensure_mesh_link() -> Path:
    """Create the repository mesh link and return its path."""
    target = assets.cache_root() / _CACHE_RESOURCE_PREFIX
    if _MESH_LINK.exists() and not _MESH_LINK.is_symlink():
        raise RuntimeError(f"Cannot create mesh link because '{_MESH_LINK}' is not a symlink")
    if _MESH_LINK.is_symlink() and _MESH_LINK.resolve() != target.resolve():
        _MESH_LINK.unlink()
    if not _MESH_LINK.is_symlink():
        _MESH_LINK.symlink_to(target, target_is_directory=True)
    return _MESH_LINK


def _relative_mesh_path(resource: str, destination_directory: Path, mesh_link: Path) -> str:
    """Resolve a mesh resource relative to the URDF directory."""
    if resource.startswith(("package://", "package:/")):
        return resource

    if resource.startswith(_CACHE_RESOURCE_PREFIX):
        linked_path = mesh_link / resource.removeprefix(_CACHE_RESOURCE_PREFIX)
    elif resource.startswith("file://"):
        linked_path = Path(resource.removeprefix("file://"))
    else:
        linked_path = Path(resource)

    return Path(operating_system_path.relpath(linked_path, destination_directory)).as_posix()


def _format_rpy(rpy: list[float]) -> str:
    """Format roll, pitch, and yaw for a URDF origin."""
    return " ".join(f"{value:.12g}" for value in rpy)


def _apply_source_rotations(robot: Robot, root: ET.Element) -> None:
    """Replace lossy Microduck frame rotations with the upstream MJCF quaternions."""
    if not robot.name.startswith("microduck"):
        return
    for joint in root.findall("joint"):
        child, origin = joint.find("child"), joint.find("origin")
        if child is None or origin is None:
            continue
        current = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        origin.set("rpy", _format_rpy(_source_body_rpy(robot.name, child.attrib["link"], current)))
    for link in root.findall("link"):
        for tag, collision in (("visual", False), ("collision", True)):
            occurrences: dict[str, int] = {}
            for element in link.findall(tag):
                mesh, origin = element.find("geometry/mesh"), element.find("origin")
                if mesh is None or origin is None:
                    continue
                stem = Path(mesh.attrib["filename"]).stem
                occurrence = occurrences.get(stem, 0)
                occurrences[stem] = occurrence + 1
                key = (link.attrib["name"], stem, collision, occurrence)
                if (quaternion := GEOM_QUATERNIONS.get(key)) is not None:
                    origin.set("rpy", _format_rpy(_quaternion_rpy(quaternion)))


def _write_urdf_with_relative_mesh_paths(robot: Robot, destination: Path) -> None:
    """Write a URDF with mesh paths relative to its destination directory."""
    mesh_link = _ensure_mesh_link()
    root = ET.fromstring(URDFGenerator().generate(robot))
    _apply_source_rotations(robot, root)
    link_names = {link.name for link in robot.links}
    joint_names = {joint.name for joint in robot.graph.joints}
    for sensor in robot.sensors:
        joint_name = f"{sensor.name}_joint"
        if sensor.name in link_names or joint_name in joint_names:
            raise ValueError(
                f"Sensor frame {sensor.name!r} conflicts with an existing link or joint"
            )
        link_names.add(sensor.name)
        joint_names.add(joint_name)
        ET.SubElement(root, "link", name=sensor.name)
        joint = ET.SubElement(root, "joint", name=joint_name, type="fixed")
        ET.SubElement(joint, "parent", link=sensor.link_name)
        ET.SubElement(joint, "child", link=sensor.name)
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(f"{value:.12g}" for value in sensor.origin.xyz),
            rpy=" ".join(f"{value:.12g}" for value in sensor.origin.rpy),
        )
    for mesh_element in root.iter("mesh"):
        if filename := mesh_element.get("filename"):
            mesh_element.set(
                "filename",
                _relative_mesh_path(filename, destination.parent.resolve(), mesh_link),
            )

    ET.indent(root)
    destination.write_text(
        ET.tostring(root, encoding="unicode", xml_declaration=True),
        encoding="utf-8",
    )


_FORMAT_BY_EXTENSION: dict[str, ExportFormat] = {
    ".blend": ExportFormat.BLENDER,
    ".glb": ExportFormat.GLB,
    ".urdf": ExportFormat.URDF,
    ".mcjf": ExportFormat.MJCF,
    ".mjcf": ExportFormat.MJCF,
}

_FORMAT_ALIASES: dict[str, ExportFormat] = {
    "blend": ExportFormat.BLENDER,
    "blender": ExportFormat.BLENDER,
    "glb": ExportFormat.GLB,
    "urdf": ExportFormat.URDF,
    "mcjf": ExportFormat.MJCF,
    "mujoco": ExportFormat.MJCF,
    "mjcf": ExportFormat.MJCF,
}


def parse_export_format(value: str) -> ExportFormat:
    """Parse an export format name.

    Args:
        value: Command-line format name.

    Returns:
        The corresponding export format.

    Raises:
        ValueError: If the format name is not supported.
    """
    if export_format := _FORMAT_ALIASES.get(value.casefold()):
        return export_format
    supported = ", ".join(export_format.value for export_format in ExportFormat)
    raise ValueError(f"unsupported format '{value}' (choose from {supported})")


def resolve_export_format(output: str | Path, export_format: ExportFormat | None) -> ExportFormat:
    """Resolve an explicit or destination-derived export format.

    Args:
        output: Destination file path.
        export_format: Optional explicit export format.

    Returns:
        The explicit format, an extension-derived format, or URDF.
    """
    return (
        export_format
        if export_format is not None
        else _FORMAT_BY_EXTENSION.get(Path(output).suffix.casefold(), ExportFormat.URDF)
    )


def export_model(
    robot: Robot,
    output: str | Path,
    export_format: ExportFormat | None = None,
    *,
    mjcf_metadata: MjcfMetadata | None = None,
    simplify_ratio: float | None = None,
) -> Path:
    """Export a robot in an explicit or destination-derived format.

    Args:
        robot: Robot to export.
        output: Destination file path.
        export_format: Optional explicit format. Infer it from `output`, falling
            back to URDF, when omitted.
        mjcf_metadata: Optional MuJoCo-specific data for MJCF export.
        simplify_ratio: Optional fraction of STL mesh faces to retain in GLB output.

    Returns:
        The written output path.
    """
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = resolve_export_format(destination, export_format)
    if simplify_ratio is not None and resolved_format is not ExportFormat.GLB:
        raise GltfExportError("Mesh simplification is only supported for GLB export")
    match resolved_format:
        case ExportFormat.BLENDER:
            export_blender(robot, destination)
        case ExportFormat.GLB:
            export_glb(robot, destination, simplify_ratio=simplify_ratio)
        case ExportFormat.MJCF:
            export_mjcf(robot, destination, mjcf_metadata=mjcf_metadata)
        case ExportFormat.URDF:
            _write_urdf_with_relative_mesh_paths(robot, destination)
    return destination


__all__ = [
    "BlenderExportError",
    "ExportFormat",
    "export_model",
    "parse_export_format",
    "resolve_export_format",
]
