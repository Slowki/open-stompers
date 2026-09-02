"""Tests for inverse dynamics and summary metrics."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from open_stompers_animations.analysis.inverse_dynamics import inverse_dynamics, support_margin
from open_stompers_animations.analysis.metrics import (
    analysis_paths,
    analyze_animation,
    read_dynamics,
)
from open_stompers_animations.format import write_animation
from open_stompers_animations.kinematics import InverseKinematics, standing_target
from open_stompers_animations.model import build_robot_model
from open_stompers_animations.types import Animation, AnimationKind, Behavior

from .conftest import MINI_PI


def _standing(frames: int = 10) -> Animation:
    """Hold the solved standing pose still."""
    robot_model = build_robot_model(MINI_PI)
    configuration = (
        InverseKinematics(robot_model)
        .solve(standing_target(robot_model), robot_model.home_configuration)
        .configuration
    )
    return Animation(
        name="standing",
        variant=str(MINI_PI),
        kind=AnimationKind.ACTION,
        sample_interval=0.02,
        joint_names=robot_model.joint_names,
        root_positions=np.tile(configuration[:3], (frames, 1)).astype(np.float32),
        root_quaternions=np.tile(configuration[3:7], (frames, 1)).astype(np.float32),
        joint_positions=np.tile(
            configuration[robot_model.joint_position_addresses], (frames, 1)
        ).astype(np.float32),
        commands=np.zeros((frames, 3), np.float32),
        behaviors=(Behavior.STAND,) * frames,
        phases=np.zeros(frames, np.float32),
        contact_names=robot_model.robot.contact_names,
        contacts=np.tile([True, True, False], (frames, 1)),
        center_of_mass=np.zeros((frames, 3), np.float32),
        zero_moment_point=np.zeros((frames, 2), np.float32),
    )


def test_standing_is_carried_entirely_by_the_feet() -> None:
    animation = _standing()
    dynamics = inverse_dynamics(animation)
    robot_model = build_robot_model(MINI_PI)
    weight = robot_model.mass * 9.81
    np.testing.assert_allclose(dynamics.base_residual, 0.0, atol=1e-3)
    np.testing.assert_allclose(dynamics.contact_forces[:, :2, 2].sum(axis=1), weight, rtol=1e-3)
    # Symmetric stance shares the weight evenly.
    np.testing.assert_allclose(
        dynamics.contact_forces[:, 0, 2], dynamics.contact_forces[:, 1, 2], rtol=0.05
    )
    assert np.abs(dynamics.joint_torques).max() < robot_model.effort_limits.min()


def test_unsupported_motion_leaves_a_residual() -> None:
    animation = _standing()
    airborne = replace(animation, contacts=np.zeros_like(animation.contacts))
    dynamics = inverse_dynamics(airborne)
    weight = build_robot_model(MINI_PI).mass * 9.81
    np.testing.assert_allclose(dynamics.base_residual[:, 2], weight, rtol=1e-3)


def test_analysis_files_round_trip(duck_walk: Animation, tmp_path: Path) -> None:
    path = write_animation(duck_walk, tmp_path / "walk.parquet")
    metrics, dynamics = analyze_animation(duck_walk, path)
    summary_path, frames_path = analysis_paths(path)
    summary = json.loads(summary_path.read_text())
    assert summary["limiting_joint"] == metrics.limiting_joint
    assert set(summary["joints"]) == set(duck_walk.joint_names)
    restored = read_dynamics(frames_path)
    np.testing.assert_allclose(restored.joint_torques, dynamics.joint_torques)
    np.testing.assert_allclose(restored.contact_forces, dynamics.contact_forces)


@pytest.mark.parametrize(
    ("point", "expected"), [((0.0, 0.0), 1.0), ((0.5, 0.0), 0.5), ((2.0, 0.0), -1.0)]
)
def test_support_margin_is_signed(point: tuple[float, float], expected: float) -> None:
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert support_margin(np.asarray(point), square) == pytest.approx(expected)
