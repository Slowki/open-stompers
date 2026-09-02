"""Tests for steady-state locomotion loops."""

import numpy as np
import pytest

from open_stompers_animations.analysis.inverse_dynamics import inverse_dynamics
from open_stompers_animations.analysis.metrics import summarize
from open_stompers_animations.generators import loops
from open_stompers_animations.model import build_robot_model
from open_stompers_animations.planning.cadence import scaled_cadence
from open_stompers_animations.robots import parse_variant
from open_stompers_animations.types import Animation, AnimationKind, Behavior, Gait

from .conftest import MICRODUCK, MINI_PI


@pytest.mark.parametrize(
    ("variant", "generator", "arguments", "command"),
    [
        (MINI_PI, loops.walk, {"velocity": 0.25}, (0.25, 0.0, 0.0)),
        (MINI_PI, loops.side_step, {"lateral_velocity": 0.08}, (0.0, 0.08, 0.0)),
        (MINI_PI, loops.turn_while_walking, {"velocity": 0.2, "yaw_rate": 0.3}, (0.2, 0.0, 0.3)),
        (MINI_PI, loops.turn_in_place, {"yaw_rate": 0.5}, (0.0, 0.0, 0.5)),
        (MINI_PI, loops.jog, {"velocity": 0.4}, (0.4, 0.0, 0.0)),
        (MICRODUCK, loops.walk, {"velocity": 0.08}, (0.08, 0.0, 0.0)),
        (MICRODUCK, loops.run, {"velocity": 0.25}, (0.25, 0.0, 0.0)),
    ],
)
def test_loop_labels_and_average_velocity_match(
    variant: object, generator: object, arguments: dict, command: tuple
) -> None:
    animation = generator(variant, **arguments)  # ty: ignore[call-non-callable]
    assert animation.kind is AnimationKind.LOOP
    np.testing.assert_allclose(animation.commands, np.tile(command, (animation.frame_count, 1)))
    # Over a whole cycle the base moves exactly as commanded.
    offset = animation.loop_offset
    assert offset.yaw / animation.duration == pytest.approx(command[2], abs=1e-6)
    if command[2] == 0:
        velocity = np.array([offset.x, offset.y]) / animation.duration
        np.testing.assert_allclose(velocity, command[:2], rtol=0.02, atol=1e-4)
    metrics = summarize(animation, inverse_dynamics(animation))
    np.testing.assert_allclose(metrics.mean_velocity[:2], command[:2], rtol=0.02, atol=2e-3)


def test_default_cadence_follows_scaled_human_data(pi_walk: Animation) -> None:
    height = build_robot_model(MINI_PI).height
    expected = scaled_cadence(Gait.WALK, 0.25, height)
    assert pi_walk.parameters["requested_cadence"] == pytest.approx(expected)
    cycle = 120 / expected / pi_walk.sample_interval
    assert abs(pi_walk.frame_count - cycle) <= 0.5


def test_running_leans_forward_with_bent_elbows() -> None:
    animation = loops.run(MINI_PI)
    quaternions = animation.root_quaternions.astype(np.float64)
    pitch = np.arcsin(
        2 * (quaternions[:, 0] * quaternions[:, 2] - quaternions[:, 3] * quaternions[:, 1])
    )
    assert pitch.mean() > 0.15
    elbow = animation.joint_positions[:, animation.joint_names.index("l_elbow_joint")]
    assert np.abs(elbow).min() > 1.0


def test_cadence_is_quantized_to_whole_frames() -> None:
    animation = loops.walk(MICRODUCK, 0.08, cadence=180)
    assert animation.parameters["requested_cadence"] == 180
    cadence = animation.parameters["cadence"]
    assert isinstance(cadence, float)
    cycle_frames = 120 / cadence / animation.sample_interval
    assert cycle_frames == pytest.approx(round(cycle_frames))
    assert animation.frame_count == round(cycle_frames)


def test_higher_cadence_shortens_the_cycle() -> None:
    slow = loops.walk(MINI_PI, 0.2, cadence=80)
    fast = loops.walk(MINI_PI, 0.2, cadence=120)
    assert fast.frame_count < slow.frame_count
    assert slow.loop_offset.x == pytest.approx(0.2 * slow.duration)


@pytest.mark.parametrize("animation_name", ["pi_walk", "duck_walk"])
def test_walking_is_dynamically_consistent(
    animation_name: str, request: pytest.FixtureRequest
) -> None:
    animation = request.getfixturevalue(animation_name)
    dynamics = inverse_dynamics(animation)
    metrics = summarize(animation, dynamics)
    # The planned zero moment point stays inside the feet, so contacts carry nearly
    # everything; rolling onto the toe before liftoff leaves a small remainder.
    assert metrics.steady_residual_ratio < 0.05
    assert metrics.minimum_support_margin > -1e-4
    assert metrics.maximum_foot_slip < 0.02
    assert metrics.center_of_mass.planned_deviation < 1e-3
    assert all(behavior is Behavior.WALK for behavior in animation.behaviors)


def test_running_has_flight_phases() -> None:
    animation = loops.run(MICRODUCK, 0.25)
    airborne = ~animation.contacts[:, :2].any(axis=1)
    assert airborne.any()
    assert animation.contacts[:, :2].any(axis=1).any()


@pytest.mark.parametrize("animation_name", ["pi_walk", "duck_walk"])
def test_joint_limits_are_respected(animation_name: str, request: pytest.FixtureRequest) -> None:
    animation = request.getfixturevalue(animation_name)
    robot_model = build_robot_model(parse_variant(animation.variant))
    columns = [robot_model.joint_index(name) for name in animation.joint_names]
    assert (animation.joint_positions >= robot_model.lower_limits[columns] - 1e-4).all()
    assert (animation.joint_positions <= robot_model.upper_limits[columns] + 1e-4).all()


def test_unreachable_speed_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        loops.walk(MINI_PI, 1.5)


def test_arms_swing_opposite_to_the_legs(pi_walk: Animation) -> None:
    left = pi_walk.joint_positions[:, pi_walk.joint_names.index("l_shoulder_pitch_joint")]
    right = pi_walk.joint_positions[:, pi_walk.joint_names.index("r_shoulder_pitch_joint")]
    assert np.ptp(left) > 0.05
    assert np.corrcoef(left, right)[0, 1] < 0


def test_walking_straightens_the_stance_leg(pi_walk: Animation) -> None:
    metrics = summarize(pi_walk, inverse_dynamics(pi_walk))
    # Home crouches the knees by 0.4 rad; walking vaults over nearly straight legs.
    assert metrics.stance_knee_flexion_minimum < 0.3
    assert metrics.stance_knee_flexion_mean < 0.8


def test_walking_uses_the_gait_step_width(pi_walk: Animation) -> None:
    dynamics = inverse_dynamics(pi_walk)
    both = pi_walk.contacts[:, :2].all(axis=1)
    lateral = np.abs(np.diff(dynamics.sole_positions[both, :, 1], axis=1))
    robot_model = build_robot_model(MINI_PI)
    # Without its own step width a gait keeps the feet under the hips.
    width = robot_model.robot.gaits[Gait.WALK].step_width or robot_model.stance_width
    np.testing.assert_allclose(lateral, width, atol=0.005)


def test_sway_stays_well_inside_the_feet(pi_walk: Animation, duck_walk: Animation) -> None:
    for animation in (pi_walk, duck_walk):
        metrics = summarize(animation, inverse_dynamics(animation))
        width = build_robot_model(parse_variant(animation.variant)).stance_width
        assert metrics.center_of_mass.lateral_sway < 0.55 * width
