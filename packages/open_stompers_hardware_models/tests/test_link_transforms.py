"""Tests for hardware-adapter link transform generation."""

import math

import pytest
from linkforge.core import RobotBuilder

from open_stompers_hardware_models import VARIANTS, MiniPiPlusVariant
from open_stompers_hardware_models.cli import main
from open_stompers_hardware_models.link_transforms import (
    _variant_model,
    adapter_crate,
    build_variant_model,
    check_link_transforms,
    sample_joint_positions,
    world_poses,
)
from open_stompers_hardware_models.variants import MicroduckVariant


def test_every_variant_maps_to_its_hardware_adapter() -> None:
    crates = {adapter_crate(variant) for variant in VARIANTS}
    assert crates == {"open-stompers-mini-pi", "open-stompers-microduck"}
    mini_pi = MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITH_WAIST
    assert adapter_crate(mini_pi) == "open-stompers-mini-pi"
    assert adapter_crate(MicroduckVariant.LEGGED) == "open-stompers-microduck"


def test_models_cover_every_link_and_use_enum_indexes() -> None:
    for variant in VARIANTS:
        model = build_variant_model(variant)
        assert model.module_name.isidentifier()
        assert len(model.links) > 1
        assert model.links[0].parent_index is None
        assert all(link.parent_index is not None for link in model.links[1:])
        assert all(
            link.parent_index is not None and link.parent_index < index
            for index, link in enumerate(model.links)
            if index > 0
        )


def test_mini_pi_plus_joint_position_limits_are_preserved() -> None:
    variant = MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITH_WAIST
    model = build_variant_model(variant)
    limits_by_name = {joint.name: joint.position_limits for joint in model.joints}

    assert len(limits_by_name) == 23
    assert all(limits is not None for limits in limits_by_name.values())
    assert limits_by_name["l_hip_roll_joint"] == pytest.approx((-0.17, 3.14))
    assert limits_by_name["r_hip_roll_joint"] == pytest.approx((-3.14, 0.17))
    assert limits_by_name["waist_yaw_joint"] == pytest.approx((-2.68, 2.68))
    assert limits_by_name["head_yaw_joint"] == pytest.approx((-1.43, 1.43))
    assert limits_by_name["head_pitch_joint"] == pytest.approx((-1.23, 1.01))


def test_zero_pose_places_children_of_the_root_at_their_joint_origins() -> None:
    model = build_variant_model(MicroduckVariant.LEGGED)
    world = world_poses(model, [0.0] * len(model.joints))
    for link in model.links:
        if link.parent_index != 0 or link.incoming is None:
            continue
        assert world[model.links.index(link)].translation == pytest.approx(
            link.incoming.origin.translation
        )


def test_joint_motion_changes_a_child_pose() -> None:
    model = build_variant_model(MicroduckVariant.LEGGED)
    zeros = world_poses(model, [0.0] * len(model.joints))
    moved = world_poses(model, sample_joint_positions(model))
    assert zeros[1].quaternion_xyzw != moved[1].quaternion_xyzw


def test_mimic_joint_is_derived_from_the_independent_joint() -> None:
    builder = RobotBuilder("mimic_toy")
    builder.link("base").root()
    (
        builder.link("upper", parent="base", joint_name="hinge")
        .revolute(axis=(0.0, 0.0, 1.0), limits=(-1.0, 1.0), xyz=(0.2, 0.0, 0.0))
        .commit()
    )
    (
        builder.link("lower", parent="upper", joint_name="copied")
        .revolute(axis=(0.0, 0.0, 1.0), limits=(-1.0, 1.0), xyz=(0.0, 0.1, 0.0))
        .mimic("hinge", multiplier=2.0, offset=0.25)
        .commit()
    )
    robot = builder.build(validate=False)
    variant = MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITHOUT_WAIST
    model = _variant_model(variant, robot)

    assert [joint.name for joint in model.joints] == ["hinge"]
    copied = model.links[-1].incoming
    assert copied is not None
    assert copied.position == "2.0 * positions.hinge + 0.25"
    moved = world_poses(model, [0.5])
    hinge = 0.5
    copied_angle = 2.0 * hinge + 0.25
    assert moved[2].quaternion_xyzw == pytest.approx(
        (0.0, 0.0, math.sin((hinge + copied_angle) / 2.0), math.cos((hinge + copied_angle) / 2.0))
    )


def test_generated_sources_match_the_workspace() -> None:
    assert check_link_transforms() == ()
    assert main(["generate-transforms", "--check"]) == 0
