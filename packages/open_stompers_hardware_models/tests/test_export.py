"""Tests for hardware model export dispatch."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from linkforge.core import InertiaTensor, RobotBuilder, box

from open_stompers_hardware_models import (
    MODELS,
    HardwareVariant,
    MiniPiPlusVariant,
    assets,
    mjcf_metadata_for,
)
from open_stompers_hardware_models.export import ExportFormat, export_model, resolve_export_format
from open_stompers_hardware_models.robots.microduck import MicroduckVariant
from open_stompers_hardware_models.robots.microduck.transforms import (
    BODY_QUATERNIONS,
    GEOM_QUATERNIONS,
)
from open_stompers_hardware_models.robots.mini_pi_plus.assets import mesh_entries


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("robot.blend", ExportFormat.BLENDER),
        ("robot.GLB", ExportFormat.GLB),
        ("robot.urdf", ExportFormat.URDF),
        ("robot.mjcf", ExportFormat.MJCF),
        ("robot.mcjf", ExportFormat.MJCF),
        ("robot.xml", ExportFormat.URDF),
    ],
)
def test_resolve_export_format_from_extension(filename: str, expected: ExportFormat) -> None:
    assert resolve_export_format(filename, None) is expected


def test_explicit_export_format_overrides_extension() -> None:
    assert resolve_export_format("robot.glb", ExportFormat.URDF) is ExportFormat.URDF


def test_export_model_defaults_to_urdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("open_stompers_hardware_models.export._MESH_LINK", tmp_path / "data")
    builder = RobotBuilder("test_robot")
    builder.link("base").mass(1.0, inertia=InertiaTensor(ixx=1.0, iyy=1.0, izz=1.0)).root()
    output = export_model(builder.build(), tmp_path / "robot.unknown")

    assert ET.parse(output).getroot().tag == "robot"


def test_exported_urdf_uses_mesh_link_relative_to_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(assets.CACHE_ROOT_ENV_VAR, str(tmp_path / "cache"))
    mesh_link = tmp_path / "data"
    monkeypatch.setattr("open_stompers_hardware_models.export._MESH_LINK", mesh_link)
    variant = MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITHOUT_WAIST
    robot = MODELS[variant]()
    for entry in mesh_entries(variant):
        mesh_path = assets.resource_path(entry.resource)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(b"mesh")
    output = export_model(
        robot,
        tmp_path / "nested" / "robot.urdf",
    )

    mesh_elements = ET.parse(output).getroot().iter("mesh")
    mesh_paths = [mesh.attrib["filename"] for mesh in mesh_elements]

    assert mesh_paths
    assert all(mesh_path.startswith("../data/") for mesh_path in mesh_paths)
    assert mesh_link.is_symlink()
    assert mesh_link.resolve() == (tmp_path / "cache" / "hardware_models").resolve()
    assert all((output.parent / mesh_path).is_file() for mesh_path in mesh_paths)


def test_export_model_writes_mjcf(tmp_path: Path) -> None:
    builder = RobotBuilder("test_robot")
    builder.link("base").visual(box(1.0, 2.0, 3.0))
    output = export_model(builder.build(), tmp_path / "robot.mjcf")

    root = ET.parse(output).getroot()
    assert root.tag == "mujoco"
    assert (geometry := root.find("./worldbody/body/geom")) is not None
    assert geometry.attrib["size"] == "0.5 1 1.5"


@pytest.mark.parametrize("variant", MODELS)
def test_exported_models_load_in_mujoco(variant: HardwareVariant, tmp_path: Path) -> None:
    """Export every registered model and load its MJCF in MuJoCo."""
    output = export_model(
        MODELS[variant](),
        tmp_path / f"{variant}.mjcf",
        ExportFormat.MJCF,
        mjcf_metadata=mjcf_metadata_for(variant),
    )

    mujoco.MjModel.from_xml_path(str(output))  # ty: ignore[unresolved-attribute]


def test_microduck_export_preserves_upstream_transforms(tmp_path: Path) -> None:
    """Preserve the upstream body and mesh frames in the Microduck MJCF."""
    output = export_model(
        MODELS[MicroduckVariant.LEGGED](),
        tmp_path / "microduck.mjcf",
        ExportFormat.MJCF,
        mjcf_metadata=mjcf_metadata_for(MicroduckVariant.LEGGED),
    )
    root = ET.parse(output).getroot()
    mesh_files = {mesh.attrib["name"]: mesh.attrib["file"] for mesh in root.findall("./asset/mesh")}

    for body in root.findall("./worldbody//body"):
        body_name = body.attrib["name"]
        if body_name in BODY_QUATERNIONS:
            assert [float(value) for value in body.attrib["quat"].split()] == pytest.approx(
                BODY_QUATERNIONS[body_name]
            )
        occurrences: dict[tuple[str, bool], int] = {}
        for geometry in body.findall("geom"):
            if "mesh" not in geometry.attrib:
                continue
            mesh_name = Path(mesh_files[geometry.attrib["mesh"]]).stem
            collision = "friction" in geometry.attrib
            occurrence_key = (mesh_name, collision)
            occurrence = occurrences.get(occurrence_key, 0)
            occurrences[occurrence_key] = occurrence + 1
            expected = GEOM_QUATERNIONS.get((body_name, mesh_name, collision, occurrence))
            if expected is not None:
                assert [float(value) for value in geometry.attrib["quat"].split()] == pytest.approx(
                    expected
                )


@pytest.mark.parametrize("variant", MODELS)
def test_sensor_frames_in_both_exports(
    variant: HardwareVariant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = MODELS[variant]()
    monkeypatch.setattr("open_stompers_hardware_models.export._MESH_LINK", tmp_path / "data")
    document = ET.parse(export_model(robot, tmp_path / "robot.urdf")).getroot()
    model = mujoco.MjModel.from_xml_path(  # ty: ignore[unresolved-attribute]
        str(
            export_model(
                robot,
                tmp_path / "robot.mjcf",
                mjcf_metadata=mjcf_metadata_for(variant),
            )
        )
    )
    for sensor in robot.sensors:
        assert document.find(f"link[@name='{sensor.name}']") is not None
        joint = document.find(f"joint[@name='{sensor.name}_joint']")
        assert joint is not None
        assert joint.attrib["type"] == "fixed"
        parent, child = joint.find("parent"), joint.find("child")
        assert parent is not None
        assert parent.attrib["link"] == sensor.link_name
        assert child is not None
        assert child.attrib["link"] == sensor.name
        origin = joint.find("origin")
        assert origin is not None
        assert [float(value) for value in origin.attrib["xyz"].split()] == pytest.approx(
            tuple(sensor.origin.xyz)
        )
        assert [float(value) for value in origin.attrib["rpy"].split()] == pytest.approx(
            tuple(sensor.origin.rpy)
        )
        site = model.site(sensor.name)
        assert model.body(int(site.bodyid[0])).name == sensor.link_name
        assert site.pos == pytest.approx(tuple(sensor.origin.xyz))
        if (camera := sensor.camera_info) is not None:
            exported_camera = model.camera(sensor.name)
            assert exported_camera.pos == pytest.approx(tuple(sensor.origin.xyz))
            assert model.cam_resolution[exported_camera.id] == pytest.approx(
                (camera.width, camera.height)
            )
            assert exported_camera.fovy[0] == pytest.approx(
                math.degrees(
                    2
                    * math.atan(math.tan(camera.horizontal_fov / 2) * camera.height / camera.width)
                )
            )
            site_rotation = np.empty(9)
            camera_rotation = np.empty(9)
            mujoco.mju_quat2Mat(site_rotation, site.quat)  # ty: ignore[unresolved-attribute]
            mujoco.mju_quat2Mat(camera_rotation, exported_camera.quat)  # ty: ignore[unresolved-attribute]
            np.testing.assert_allclose(
                camera_rotation.reshape(3, 3),
                site_rotation.reshape(3, 3) @ np.diag((1, -1, -1)),
                atol=1e-10,
            )
        if sensor.imu_info is not None:
            for suffix in ("ang_vel", "accel"):
                assert model.sensor(f"{sensor.name}_{suffix}").objid[0] == site.id


@pytest.mark.parametrize("variant", MODELS)
def test_imu_placement_matches_source(variant: HardwareVariant) -> None:
    sensors = {sensor.name: sensor for sensor in MODELS[variant]().sensors}
    inertial_sensor = sensors["imu"]
    if isinstance(variant, MicroduckVariant):
        assert inertial_sensor.link_name == "trunk_base"
        assert tuple(inertial_sensor.origin.xyz) == (-0.021, 0.0000664098, -0.0146984)
        head_sensor = sensors["head_imu"]
        assert head_sensor.link_name == "jaw_soft"
        assert tuple(head_sensor.origin.xyz) == (0.0152323, 0.000111069, -0.05106)
    else:
        assert inertial_sensor.link_name == "base_link"
        assert tuple(inertial_sensor.origin.xyz) == (0.0, 0.0, 0.0)
    assert tuple(inertial_sensor.origin.rpy) == (0.0, 0.0, 0.0)
