"""Tests for one-shot action generators."""

import numpy as np
import pytest

from open_stompers_animations.analysis.inverse_dynamics import inverse_dynamics
from open_stompers_animations.analysis.metrics import summarize
from open_stompers_animations.generators import GENERATORS, actions
from open_stompers_animations.types import AnimationKind, Behavior

from .conftest import MICRODUCK, MINI_PI

# Stronger tests already generate these pairs and check more than finiteness.
_ALREADY_COVERED = {
    ("walk", MINI_PI),
    ("walk", MICRODUCK),
    ("jog", MINI_PI),
    ("run", MINI_PI),
    ("run", MICRODUCK),
    ("turn_in_place", MINI_PI),
    ("turn_while_walking", MINI_PI),
    ("side_step", MINI_PI),
    ("start_walking", MINI_PI),
    ("sit_down", MINI_PI),
    ("sit_down", MICRODUCK),
    ("stand_up", MICRODUCK),
    ("jump_vertical", MICRODUCK),
    ("run_to_stop", MICRODUCK),
}


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        (name, variant)
        for name in sorted(GENERATORS)
        for variant in (MINI_PI, MICRODUCK)
        if (name, variant) not in _ALREADY_COVERED
    ],
)
def test_every_generator_runs(name: str, variant: object) -> None:
    animation = GENERATORS[name](variant)
    assert animation.frame_count > 10
    assert np.isfinite(animation.joint_positions).all()


def test_jump_has_a_ballistic_flight() -> None:
    animation = actions.jump_vertical(MICRODUCK)
    airborne = ~animation.contacts[:, :2].any(axis=1)
    assert airborne.sum() >= 3
    dynamics = inverse_dynamics(animation)
    inner = np.flatnonzero(airborne)[1:-1]
    np.testing.assert_allclose(dynamics.contact_forces[inner], 0.0, atol=1e-6)
    assert Behavior.JUMP in animation.behaviors


def test_stand_up_reverses_sit_down() -> None:
    seated = actions.sit_down(MICRODUCK)
    rising = actions.stand_up(MICRODUCK)
    np.testing.assert_array_equal(rising.joint_positions, seated.joint_positions[::-1])
    assert rising.behaviors[0] is Behavior.SIT
    assert rising.behaviors[-1] is Behavior.STAND
    assert Behavior.STAND_UP in rising.behaviors


def test_mini_pi_sits_on_its_pad() -> None:
    animation = actions.sit_down(MINI_PI)
    pad = animation.contact_names.index("support_pad")
    assert animation.contacts[-1, pad]
    assert not animation.contacts[0, pad]


def test_run_to_stop_starts_running_and_ends_at_rest() -> None:
    animation = actions.run_to_stop(MICRODUCK)
    assert animation.kind is AnimationKind.ACTION
    assert animation.behaviors[0] is Behavior.RUN
    assert animation.behaviors[-1] is Behavior.STAND
    np.testing.assert_allclose(animation.commands[-1], 0.0)
    assert np.linalg.norm(animation.base_linear_velocities[-5:], axis=1).max() < 0.01
    # The kept window is moved to start at the origin.
    np.testing.assert_allclose(animation.center_of_mass[0, :2], 0.0, atol=1e-6)


def test_start_walking_begins_standing() -> None:
    animation = actions.start_walking(MINI_PI)
    assert animation.behaviors[0] is Behavior.STAND
    assert animation.behaviors[-1] is Behavior.WALK
    metrics = summarize(animation, inverse_dynamics(animation))
    assert metrics.steady_residual_ratio < 0.05
