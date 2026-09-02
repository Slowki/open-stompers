"""Structural and asset tests for the Microduck LinkForge models."""

from pathlib import Path

import numpy as np
import pytest
from linkforge.core import JointType, Mesh, URDFGenerator, validate_robot

from open_stompers_hardware_models import load, mjcf_metadata_for
from open_stompers_hardware_models.mjcf import export_mjcf
from open_stompers_hardware_models.robots import microduck
from open_stompers_hardware_models.robots.microduck import MicroduckVariant, assets

EXPECTED_STRUCTURE = {
    MicroduckVariant.LEGGED: (15, 14, 14),
    MicroduckVariant.ROLLER_SKATES: (19, 18, 18),
}


@pytest.mark.parametrize("variant", microduck.VARIANTS)
def test_structure(variant: MicroduckVariant) -> None:
    robot = load(variant)
    links, joints, movable = EXPECTED_STRUCTURE[variant]
    assert len(robot.graph.link_names) == links
    assert len(list(robot.graph.joints)) == joints
    assert robot.degrees_of_freedom == movable
    assert robot.root_link.name == "trunk_base"


@pytest.mark.parametrize("variant", microduck.VARIANTS)
def test_model_is_valid_and_roundtrips_to_urdf(variant: MicroduckVariant) -> None:
    robot = load(variant)
    result = validate_robot(robot)
    assert result.is_valid
    assert not result.errors
    assert f'<robot name="{variant}"' in URDFGenerator().generate(robot)


def test_rollers_have_passive_wheel_joints() -> None:
    robot = load(MicroduckVariant.ROLLER_SKATES)
    wheels = {
        joint.name: joint
        for joint in robot.graph.joints
        if joint.name.startswith("passive_") and joint.name.endswith("_wheel")
    }
    assert set(wheels) == {
        "passive_LF_wheel",
        "passive_LR_wheel",
        "passive_RF_wheel",
        "passive_RR_wheel",
    }
    for joint in wheels.values():
        assert joint.type is JointType.CONTINUOUS
        assert joint.limits is not None
        assert joint.limits.lower is None
        assert joint.limits.upper is None


@pytest.mark.parametrize("variant", microduck.VARIANTS)
def test_camera_and_time_of_flight_sensors(variant: MicroduckVariant) -> None:
    robot = load(variant)
    sensors = {sensor.name: sensor for sensor in robot.sensors}

    assert {"head_camera", "tof"} <= sensors.keys()
    camera = sensors["head_camera"]
    assert camera.camera_info is not None
    assert camera.link_name == "jaw_soft"
    assert camera.origin.xyz.to_tuple() == pytest.approx((0.0155, -0.0000913778, -0.0733))

    tof = sensors["tof"]
    assert tof.type.value == "depth_camera"
    assert tof.link_name == "jaw_soft"
    assert tof.camera_info is not None
    assert tof.camera_info.horizontal_fov == pytest.approx(0.7853981633974483)
    assert (tof.camera_info.width, tof.camera_info.height) == (8, 8)
    assert tuple(tof.origin.xyz) == pytest.approx((0.0135, 0.0224086, -0.0733))


@pytest.mark.parametrize("variant", microduck.VARIANTS)
def test_mesh_resources_match_manifest(variant: MicroduckVariant) -> None:
    robot = load(variant)
    resources = set()
    for name in robot.graph.link_names:
        link = robot.get_link(name)
        assert link is not None
        for visual in link.visuals:
            assert isinstance(visual.geometry, Mesh)
            resources.add(visual.geometry.resource)
    expected = {
        assets.get_cache_relative_path(variant, filename)
        for filename in assets.MESH_MANIFESTS[variant]
    }
    assert resources <= expected


def test_mjcf_measurement_frames_and_sensors(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(
        str(
            export_mjcf(
                load(MicroduckVariant.LEGGED),
                tmp_path / "microduck.xml",
                mjcf_metadata=mjcf_metadata_for(MicroduckVariant.LEGGED),
            )
        )
    )
    assert model.nsite == 7
    assert model.nsensor == 10
    assert model.ncam == 2
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = model.camera("head_camera").id
    forward = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(forward, [1.0, 0.0, 0.0], atol=1e-6)
    assert model.body(model.site("imu").bodyid[0]).name == "trunk_base"
    np.testing.assert_allclose(model.site("imu").pos, [-0.021, 0.0000664098, -0.0146984])
    for side, body in (("left", "ankle_left"), ("right", "ankle_right")):
        assert model.body(model.site(f"{side}_foot").bodyid[0]).name == body
    assert int(model.sensor("root_angmom").type[0]) == int(mujoco.mjtSensor.mjSENS_SUBTREEANGMOM)
    assert int(model.sensor("imu_ang_vel").type[0]) == int(mujoco.mjtSensor.mjSENS_GYRO)
