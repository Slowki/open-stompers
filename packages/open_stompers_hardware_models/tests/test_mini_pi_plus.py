"""Structural tests for the Mini Pi Plus LinkForge models."""

import math
from typing import cast

import pytest
from linkforge.core import Mesh

from open_stompers_hardware_models import HardwareVariant, load
from open_stompers_hardware_models.robots import common, mini_pi_plus
from open_stompers_hardware_models.robots.mini_pi_plus import (
    MiniPiPlusVariant,
    assets,
    piplus_s_12l8a0g2h0w as second_variant,
    piplus_s_12l8a0g2h1w as third_variant,
    piplus_s_12l10a0g0h1w as first_variant,
)
from open_stompers_hardware_models.robots.mini_pi_plus.materials import (
    ALUMINIUM_MATERIAL_NAME,
    ALUMINIUM_RGBA,
    BLACK_ABS_MATERIAL_NAME,
    BLACK_ABS_RGBA,
    GRAY_RGBA,
    MATERIAL_NAME,
)

EXPECTED_STRUCTURE: dict[MiniPiPlusVariant, tuple[int, int, int]] = {
    # (links, joints, movable joints)
    first_variant.VARIANT: (27, 26, 23),
    second_variant.VARIANT: (27, 26, 22),
    third_variant.VARIANT: (28, 27, 23),
}

ACTUATED_BY_VARIANT = {
    first_variant.VARIANT: first_variant.ACTUATED_JOINTS,
    second_variant.VARIANT: second_variant.ACTUATED_JOINTS,
    third_variant.VARIANT: third_variant.ACTUATED_JOINTS,
}


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="Unknown hardware model"):
        load(cast(HardwareVariant, "PiPlus_S_does_not_exist"))


def test_mini_pi_plus_unknown_variant_raises() -> None:
    with pytest.raises(KeyError, match="Unknown Mini Pi Plus variant"):
        mini_pi_plus.build(cast(MiniPiPlusVariant, "PiPlus_S_does_not_exist"))


def test_structure(variant: MiniPiPlusVariant) -> None:
    robot = load(variant)
    links, joints, movable = EXPECTED_STRUCTURE[variant]
    assert len(robot.graph.link_names) == links
    assert len(list(robot.graph.joints)) == joints
    assert robot.degrees_of_freedom == movable
    assert robot.root_link.name == "base_link"


def test_zed_mini_camera_sensors() -> None:
    robot = load(second_variant.VARIANT)
    sensors = {sensor.name: sensor for sensor in robot.sensors}

    assert set(sensors) == {"left_camera", "right_camera", "imu"}
    for name, y_offset in (("left_camera", 0.03), ("right_camera", -0.03)):
        sensor = sensors[name]
        assert sensor.link_name == "camera_link"
        assert sensor.camera_info is not None
        assert sensor.camera_info.horizontal_fov == pytest.approx(math.radians(102.0))
        assert (sensor.camera_info.width, sensor.camera_info.height) == (2208, 1242)
        assert sensor.origin.xyz.to_tuple() == pytest.approx((0.0, y_offset, 0.01325))


def test_actuated_joints_match_model(variant: MiniPiPlusVariant) -> None:
    robot = load(variant)
    actuated = ACTUATED_BY_VARIANT[variant]
    assert len(actuated) == len(set(actuated))
    model_movable = common.actuated_joint_names(robot)
    assert set(actuated) == set(model_movable)


def test_actuated_joints_follow_upstream_controller_order(variant: MiniPiPlusVariant) -> None:
    """The actuated-joint order mirrors the upstream controller joint list.

    For the first variant, the upstream list still names the fixed head joints
    and omits the wrists; the model order swaps them accordingly.
    """
    actuated = ACTUATED_BY_VARIANT[variant]
    match variant:
        case first_variant.VARIANT:
            assert "head_yaw_joint" not in actuated
            assert "head_pitch_joint" not in actuated
            assert actuated.index("l_wrist_joint") == actuated.index("l_elbow_joint") + 1
            assert actuated.index("r_wrist_joint") == actuated.index("r_elbow_joint") + 1
        case _:
            assert "head_yaw_joint" in actuated
            assert "head_pitch_joint" in actuated
            assert "l_wrist_joint" not in actuated
            assert "r_wrist_joint" not in actuated


def test_mesh_resources_match_manifest(variant: MiniPiPlusVariant) -> None:
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
    assert resources == expected


def test_material_assignments(variant: MiniPiPlusVariant) -> None:
    robot = load(variant)
    expected_materials = third_variant.MATERIAL_BY_LINK
    for name in robot.graph.link_names:
        link = robot.get_link(name)
        assert link is not None
        for visual in link.visuals:
            assert visual.material is not None
            expected = (
                expected_materials[name] if variant == third_variant.VARIANT else MATERIAL_NAME
            )
            assert visual.material.name == expected


def test_third_variant_material_assignments() -> None:
    robot = load(third_variant.VARIANT)
    expected_materials = {
        "base_link": ALUMINIUM_MATERIAL_NAME,
        **{
            f"{side}_{part}_link": BLACK_ABS_MATERIAL_NAME
            for side in ("l", "r")
            for part in ("shoulder_pitch", "hip_roll", "thigh", "calf", "ankle_roll")
        },
        "l_shoulder_roll_link": ALUMINIUM_MATERIAL_NAME,
        "r_shoulder_roll_link": ALUMINIUM_MATERIAL_NAME,
        "l_elbow_link": ALUMINIUM_MATERIAL_NAME,
        "r_elbow_link": ALUMINIUM_MATERIAL_NAME,
        "head_yaw_link": ALUMINIUM_MATERIAL_NAME,
    }

    for link_name, expected_material in expected_materials.items():
        link = robot.get_link(link_name)
        assert link is not None
        assert link.visuals[0].material is not None
        assert link.visuals[0].material.name == expected_material


def test_pbr_material_colors_match_export_profiles(variant: MiniPiPlusVariant) -> None:
    robot = load(variant)
    for link in robot.links:
        for visual in link.visuals:
            assert visual.material is not None
            assert visual.material.color is not None
            if visual.material.name == ALUMINIUM_MATERIAL_NAME:
                expected = ALUMINIUM_RGBA
            elif visual.material.name == BLACK_ABS_MATERIAL_NAME:
                expected = BLACK_ABS_RGBA
            else:
                expected = GRAY_RGBA
            assert visual.material.color.to_tuple() == expected
