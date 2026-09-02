"""Prepare robot scenes and manifests for Blender rig export."""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import trimesh
from linkforge.core import Mesh, Robot

from open_stompers_hardware_models.gltf import _apply_material, _geometry, _transform
from open_stompers_hardware_models.robots.microduck.transforms import (
    BODY_QUATERNIONS,
    GEOM_QUATERNIONS,
)

_LIMB_ROOTS: dict[str, dict[str, tuple[str, str]]] = {
    "PiPlus_S_12L10A0G0H1W": {
        "left_hand": ("l_shoulder_pitch_joint", "l_wrist_link"),
        "right_hand": ("r_shoulder_pitch_joint", "r_wrist_link"),
        "left_foot": ("l_hip_pitch_joint", "l_ankle_roll_link"),
        "right_foot": ("r_hip_pitch_joint", "r_ankle_roll_link"),
    },
    "PiPlus_S_12L8A0G2H0W": {
        "left_hand": ("l_shoulder_pitch_joint", "l_wrist_link"),
        "right_hand": ("r_shoulder_pitch_joint", "r_wrist_link"),
        "left_foot": ("l_hip_pitch_joint", "l_ankle_roll_link"),
        "right_foot": ("r_hip_pitch_joint", "r_ankle_roll_link"),
    },
    "PiPlus_S_12L8A0G2H1W": {
        "left_hand": ("l_shoulder_pitch_joint", "l_wrist_link"),
        "right_hand": ("r_shoulder_pitch_joint", "r_wrist_link"),
        "left_foot": ("l_hip_pitch_joint", "l_ankle_roll_link"),
        "right_foot": ("r_hip_pitch_joint", "r_ankle_roll_link"),
    },
    "microduck": {
        "left_foot": ("left_hip_yaw", "ankle_left"),
        "right_foot": ("right_hip_yaw", "ankle_right"),
    },
    "microduck_rollers": {
        "left_foot": ("left_hip_yaw", "ankle_l_v1"),
        "right_foot": ("right_hip_yaw", "ankle_r_v1"),
    },
}


class BlenderExportError(RuntimeError):
    """Blender could not create a project file."""


def _quaternion_rpy(quaternion: tuple[float, float, float, float]) -> list[float]:
    """Convert a WXYZ quaternion to XYZ/RPY angles."""
    w, x, y, z = quaternion
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    if abs(pitch_sine) >= 1.0 - 1e-8:
        pitch = math.copysign(math.pi / 2.0, pitch_sine)
        rotation_01 = 2.0 * (x * y - w * z)
        rotation_11 = 1.0 - 2.0 * (x * x + z * z)
        roll = math.atan2(pitch_sine * rotation_01, rotation_11)
        yaw = 0.0
    else:
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(pitch_sine)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def _source_body_rpy(robot_name: str, link_name: str, fallback: list[float]) -> list[float]:
    """Convert a Microduck source body quaternion to XYZ/RPY angles."""
    if not robot_name.startswith("microduck"):
        return fallback
    if (quaternion := BODY_QUATERNIONS.get(link_name)) is None:
        return fallback
    return _quaternion_rpy(quaternion)


def _source_visual_rpy(
    robot_name: str,
    link_name: str,
    geometry_name: str,
    geometry_index: int,
    fallback: list[float],
) -> list[float]:
    """Convert a Microduck source visual quaternion to XYZ/RPY angles."""
    if not robot_name.startswith("microduck"):
        return fallback

    if (
        quaternion := GEOM_QUATERNIONS.get((link_name, geometry_name, False, geometry_index))
    ) is None:
        return fallback
    return _quaternion_rpy(quaternion)


def _build_blender_scene(robot: Robot, manifest: dict[str, Any]) -> trimesh.Scene:
    """Build Blender input with visuals kept in their owning link frames."""
    scene = trimesh.Scene()
    z_up_to_gltf = _transform((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0))
    visuals = iter(manifest["visuals"])
    for link_name in robot.graph.get_topological_link_names():
        for visual in robot.link(link_name).visuals:
            mesh = _geometry(robot, visual.geometry)
            _apply_material(mesh, visual.material)
            manifest_visual = next(visuals)
            visual_transform = _transform(
                tuple(manifest_visual["xyz"]), tuple(manifest_visual["rpy"])
            )
            scene.add_geometry(
                mesh,
                transform=z_up_to_gltf @ visual_transform,
                node_name=manifest_visual["node"],
            )
    return scene


def _limb_chains(robot: Robot) -> list[dict[str, Any]]:
    """Resolve configured limb targets to their model joint chains."""
    roots = _LIMB_ROOTS.get(robot.name, {})
    joints_by_child = {joint.child: joint for joint in robot.graph.joints}
    chains: list[dict[str, Any]] = []
    for control_name, (root_joint_name, endpoint_link) in roots.items():
        root_joint = next(joint for joint in robot.graph.joints if joint.name == root_joint_name)
        chain: list[str] = []
        current_link = endpoint_link
        while current_link != root_joint.parent:
            joint = joints_by_child[current_link]
            chain.append(joint.name)
            current_link = joint.parent
        chain.reverse()
        chains.append(
            {
                "name": control_name,
                "root_joint": root_joint_name,
                "endpoint_link": endpoint_link,
                "joints": chain,
                "orientation_weight": 0.15 if robot.name.startswith("microduck") else 1.0,
            }
        )
    return chains


def build_blender_manifest(robot: Robot) -> dict[str, Any]:
    """Describe link transforms, meshes, joints, and limb targets."""
    link_names = robot.graph.get_topological_link_names()
    joint_by_child = {joint.child: joint for joint in robot.graph.joints}
    links: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    visual_indices: dict[tuple[str, str], int] = {}
    for link_name in link_names:
        link = robot.link(link_name)
        links.append(
            {
                "name": link.name,
                "parent": joint_by_child[link.name].parent if link.name in joint_by_child else None,
                "joint": joint_by_child[link.name].name if link.name in joint_by_child else None,
            }
        )
        for index, visual in enumerate(link.visuals):
            geometry_name = ""
            if isinstance(visual.geometry, Mesh):
                geometry_name = Path(visual.geometry.resource).stem
            geometry_key = (link_name, geometry_name)
            geometry_index = visual_indices.get(geometry_key, 0)
            visual_indices[geometry_key] = geometry_index + 1
            visuals.append(
                {
                    "node": f"open_stompers_visual_{len(visuals):04d}",
                    "link": link_name,
                    "name": visual.name or f"{link_name}_visual_{index}",
                    "xyz": list(visual.origin.xyz),
                    "rpy": _source_visual_rpy(
                        robot.name,
                        link_name,
                        geometry_name,
                        geometry_index,
                        list(visual.origin.rpy),
                    ),
                }
            )

    joints = [
        {
            "name": joint.name,
            "type": str(joint.type),
            "parent": joint.parent,
            "child": joint.child,
            "xyz": list(joint.origin.xyz),
            "rpy": _source_body_rpy(
                robot.name,
                joint.child,
                list(joint.origin.rpy),
            ),
            "axis": list(joint.axis) if joint.axis is not None else None,
            "lower": joint.limits.lower if joint.limits is not None else None,
            "upper": joint.limits.upper if joint.limits is not None else None,
        }
        for joint in robot.graph.get_topological_joints()
    ]
    return {
        "name": robot.name,
        "root_link": robot.root_link.name,
        "links": links,
        "joints": joints,
        "visuals": visuals,
        "limbs": _limb_chains(robot),
    }


def export_blender(robot: Robot, output: str | Path) -> Path:
    """Export a rigid-link robot rig as a native Blender project."""
    blender_executable = shutil.which("blender")
    if blender_executable is None:
        raise BlenderExportError("Blender export requires the 'blender' executable")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_blender_manifest(robot)
    with tempfile.TemporaryDirectory(prefix="open-stompers-blender-") as temporary:
        directory = Path(temporary)
        glb_path = directory / "model.glb"
        manifest_path = directory / "manifest.json"
        _build_blender_scene(robot, manifest).export(glb_path, file_type="glb")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        script_path = Path(__file__).with_name("blender_rig.py")
        try:
            process = subprocess.run(
                [
                    blender_executable,
                    "--background",
                    "--python",
                    str(script_path),
                    "--",
                    str(glb_path.resolve()),
                    str(manifest_path.resolve()),
                    str(destination.resolve()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if not destination.is_file():
                details = (
                    process.stderr.strip() or process.stdout.strip() or "no output file was saved"
                )
                raise BlenderExportError(f"Blender export failed: {details}")
        except subprocess.CalledProcessError as error:
            details = error.stderr.strip() or error.stdout.strip() or "unknown Blender error"
            raise BlenderExportError(f"Blender export failed: {details}") from error
    return destination
