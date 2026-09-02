"""Tests for nominal standing poses."""

import pytest

from open_stompers_hardware_models import VARIANTS, HardwareVariant, load
from open_stompers_hardware_models.poses import home_pose
from open_stompers_hardware_models.robots.common import actuated_joint_names


@pytest.mark.parametrize("variant", VARIANTS)
def test_home_pose_covers_every_actuated_joint(variant: HardwareVariant) -> None:
    pose = home_pose(variant)
    assert set(actuated_joint_names(load(variant))) <= set(pose)
    for joint in load(variant).graph.joints:
        limits = joint.limits
        if joint.name in pose and limits is not None:
            if limits.lower is not None:
                assert pose[joint.name] >= limits.lower
            if limits.upper is not None:
                assert pose[joint.name] <= limits.upper
