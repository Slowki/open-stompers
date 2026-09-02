"""Tests for GLB export and pose handling."""

import math
from pathlib import Path

import pytest
import trimesh
from linkforge.core import JointType, Material, Mesh, Robot, RobotBuilder, box

import open_stompers_hardware_models
from open_stompers_hardware_models import MicroduckVariant
from open_stompers_hardware_models.gltf import GltfExportError, export_all, export_glb


def _robot() -> Robot:
    """Build a small robot with one revolute joint for exporter tests."""
    builder = RobotBuilder("gltf_test")
    builder.link("base").visual(box(1.0, 1.0, 1.0))
    child = builder.link("arm", parent="base", joint_name="arm_joint")
    child.revolute((0.0, 0.0, 1.0), limits=(-math.pi, math.pi), xyz=(1.0, 0.0, 0.0))
    child.visual(box(1.0, 0.2, 0.2))
    return builder.build()


def test_export_glb_defaults_pose_to_zero(tmp_path: Path) -> None:
    output = export_glb(_robot(), tmp_path / "robot.glb")

    scene = trimesh.load_scene(output)
    assert output.read_bytes()[:4] == b"glTF"
    assert len(scene.geometry) == 2


def test_export_glb_writes_pbr_factors_and_fallbacks(tmp_path: Path) -> None:
    builder = RobotBuilder("pbr_test")
    builder.material("aluminium", color=(0.92, 0.94, 0.97, 1.0))
    builder.material("custom_diffuse", color=(0.2, 0.3, 0.4, 1.0))
    builder.link("base").visual(box(1.0, 1.0, 1.0), material="aluminium")
    builder.link("child", parent="base", joint_name="fixed").fixed().visual(
        box(0.5, 0.5, 0.5), material="custom_diffuse"
    )
    builder.link("texture_only", parent="base", joint_name="texture_only").fixed().visual(
        box(0.25, 0.25, 0.25), material=Material(name="texture_only", texture="missing.png")
    )

    scene = trimesh.load_scene(export_glb(builder.build(), tmp_path / "pbr.glb"))
    materials = {
        mesh.visual.material.name: mesh.visual.material for mesh in scene.geometry.values()
    }

    aluminium = materials["aluminium"]
    assert aluminium.metallicFactor == pytest.approx(0.92)
    assert aluminium.roughnessFactor == pytest.approx(0.24)
    fallback = materials["custom_diffuse"]
    assert fallback.metallicFactor == pytest.approx(0.0)
    assert fallback.roughnessFactor == pytest.approx(0.5)
    texture_fallback = materials["texture_only"]
    assert texture_fallback.metallicFactor == pytest.approx(0.0)
    assert texture_fallback.roughnessFactor == pytest.approx(0.5)


def test_export_glb_simplifies_stl_and_preserves_pbr_material(tmp_path: Path) -> None:
    source_mesh = trimesh.creation.icosphere(subdivisions=3)
    source_path = tmp_path / "source.stl"
    source_mesh.export(source_path)
    builder = RobotBuilder("simplification_test")
    builder.material("custom_green", color=(0.2, 0.7, 0.3, 1.0))
    builder.link("base").visual(Mesh(str(source_path)), material="custom_green")

    original_output = export_glb(builder.build(), tmp_path / "original.glb")
    output = export_glb(builder.build(), tmp_path / "simplified.glb", simplify_ratio=0.25)
    exported_mesh = next(iter(trimesh.load_scene(output).geometry.values()))

    assert len(exported_mesh.faces) < len(source_mesh.faces)
    assert output.stat().st_size < original_output.stat().st_size
    assert exported_mesh.visual.material.name == "custom_green"
    assert exported_mesh.visual.material.baseColorFactor == pytest.approx((51, 178, 76, 255))


@pytest.mark.parametrize("ratio", [-0.1, float("nan")])
def test_export_glb_rejects_invalid_simplification_ratio(tmp_path: Path, ratio: float) -> None:
    with pytest.raises(GltfExportError, match="Simplification ratio"):
        export_glb(_robot(), tmp_path / "invalid.glb", simplify_ratio=ratio)


def test_export_glb_full_retention_skips_simplification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.stl"
    source_mesh = trimesh.creation.icosphere(subdivisions=2)
    source_mesh.export(source_path)
    builder = RobotBuilder("full_retention_test")
    builder.link("base").visual(Mesh(str(source_path)))
    monkeypatch.setattr(
        trimesh.Trimesh,
        "simplify_quadric_decimation",
        lambda *_args, **_kwargs: pytest.fail("full retention must skip decimation"),
    )

    output = export_glb(builder.build(), tmp_path / "unchanged.glb", simplify_ratio=1.0)
    exported_mesh = next(iter(trimesh.load_scene(output).geometry.values()))
    assert len(exported_mesh.faces) == len(source_mesh.faces)


def test_export_glb_wraps_simplification_backend_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.stl"
    trimesh.creation.icosphere(subdivisions=2).export(source_path)
    builder = RobotBuilder("backend_error_test")
    builder.link("base").visual(Mesh(str(source_path)))
    monkeypatch.setattr(
        trimesh.Trimesh,
        "simplify_quadric_decimation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    with pytest.raises(GltfExportError, match="backend failed"):
        export_glb(builder.build(), tmp_path / "failed.glb", simplify_ratio=0.25)


def test_export_glb_applies_joint_pose(tmp_path: Path) -> None:
    robot = _robot()
    zero_path = export_glb(robot, tmp_path / "zero.glb")
    posed_path = export_glb(robot, tmp_path / "posed.glb", {"arm_joint": math.pi / 2})

    zero = trimesh.load_scene(zero_path)
    posed = trimesh.load_scene(posed_path)
    zero_bounds = sorted(tuple(mesh.bounds[:, 0]) for mesh in zero.geometry.values())
    posed_bounds = sorted(tuple(mesh.bounds[:, 0]) for mesh in posed.geometry.values())
    assert zero_bounds != posed_bounds


def test_export_glb_converts_z_up_to_gltf_y_up(tmp_path: Path) -> None:
    builder = RobotBuilder("up_axis_test")
    builder.link("base").visual(box(1.0, 1.0, 1.0), xyz=(0.0, 0.0, 2.0))

    scene = trimesh.load_scene(export_glb(builder.build(), tmp_path / "up.glb"))
    bounds = next(iter(scene.geometry.values())).bounds

    assert bounds[:, 1].mean() == pytest.approx(2.0)
    assert bounds[:, 2].mean() == pytest.approx(0.0)


def test_export_glb_rejects_unknown_joint(tmp_path: Path) -> None:
    with pytest.raises(GltfExportError, match="unknown joint"):
        export_glb(_robot(), tmp_path / "robot.glb", {"missing": 1.0})


def test_export_all_writes_one_file_per_robot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(open_stompers_hardware_models, "load", lambda _name: _robot())

    paths = export_all(
        tmp_path / "models", variants=(MicroduckVariant.LEGGED, MicroduckVariant.ROLLER_SKATES)
    )

    assert paths == [
        tmp_path / "models" / "microduck.glb",
        tmp_path / "models" / "microduck_rollers.glb",
    ]
    assert all(path.is_file() and path.read_bytes()[:4] == b"glTF" for path in paths)


def test_fixed_joint_pose_is_zero_only(tmp_path: Path) -> None:
    builder = RobotBuilder("fixed_test")
    builder.link("base")
    builder.link("child", parent="base", joint_name="fixed").fixed()
    robot = builder.build()
    assert (joint := robot.get_joint("fixed")) is not None
    assert joint.type is JointType.FIXED

    with pytest.raises(GltfExportError, match="Fixed joint"):
        export_glb(robot, tmp_path / "robot.glb", {"fixed": 1.0})
