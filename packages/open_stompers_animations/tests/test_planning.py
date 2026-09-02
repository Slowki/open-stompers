"""Tests for footstep and center of mass planning."""

import numpy as np
import pytest

from open_stompers_animations.planning.cadence import (
    HUMAN_HEIGHT,
    human_cadence,
    human_equivalent_speed,
    scaled_cadence,
)
from open_stompers_animations.planning.center_of_mass import (
    GRAVITY,
    plan_center_of_mass,
    plan_vertical,
)
from open_stompers_animations.planning.footsteps import integrate_twist
from open_stompers_animations.planning.swing import clearance, minimum_jerk
from open_stompers_animations.types import Gait, LoopOffset


@pytest.mark.parametrize("twist", [(0.3, 0.0, 0.0), (0.2, 0.1, 0.5), (0.0, 0.0, -0.8)])
def test_twist_integration_composes_exactly(twist: tuple[float, float, float]) -> None:
    whole = integrate_twist(np.zeros(3), np.asarray(twist), 1.0)[0]
    pose = np.zeros(3)
    for _ in range(10):
        pose = integrate_twist(pose, np.asarray(twist), 0.1)[0]
    np.testing.assert_allclose(pose, whole, atol=1e-12)


def test_profiles_start_and_end_at_rest() -> None:
    assert minimum_jerk([0.0, 1.0]) == pytest.approx([0.0, 1.0])
    assert clearance([0.0, 0.5, 1.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert clearance(0.3, peak=0.3) == pytest.approx(1.0)


def test_vertical_plan_free_falls_when_airborne() -> None:
    interval, count = 0.005, 200
    airborne = np.zeros(count, dtype=bool)
    airborne[80:120] = True
    heights, _, accelerations = plan_vertical(
        interval, np.full(count, 0.3), np.full(count, 1.0), airborne, periodic=True
    )
    np.testing.assert_allclose(accelerations[82:118], -GRAVITY, atol=1e-6)
    assert heights.mean() == pytest.approx(0.3, abs=0.01)


def test_center_of_mass_tracks_a_static_reference() -> None:
    interval, count = 0.005, 400
    reference = np.tile([0.05, -0.02], (count, 1))
    plan = plan_center_of_mass(
        interval, reference, np.full(count, 0.3), np.zeros(count), np.zeros(count), periodic=False
    )
    np.testing.assert_allclose(plan.zero_moment_point[50:-50], reference[50:-50], atol=1e-4)


def test_periodic_plan_respects_the_loop_offset() -> None:
    interval, count = 0.005, 240
    offset = LoopOffset(0.3, 0.0, 0.0)
    times = np.arange(count) * interval
    reference = np.stack((0.3 * times / (count * interval), np.zeros(count)), axis=1)
    plan = plan_center_of_mass(
        interval,
        reference,
        np.full(count, 0.3),
        np.zeros(count),
        np.zeros(count),
        periodic=True,
        offset=offset,
    )
    np.testing.assert_allclose(plan.velocities[:, 0], 0.25, atol=1e-3)


def test_human_cadence_matches_reference_data() -> None:
    # Normal walking near 1.3 m/s is about 110 steps per minute.
    assert human_cadence(Gait.WALK, 1.3) == pytest.approx(110, abs=3)
    # Recreational runners chose about 154 to 165 steps per minute at 2.4 to 3.0 m/s.
    assert 150 <= human_cadence(Gait.RUN, 2.4) <= 160
    assert 158 <= human_cadence(Gait.RUN, 3.0) <= 168
    assert human_cadence(Gait.RUN, 4.0) > human_cadence(Gait.RUN, 3.0)


def test_cadence_scales_by_dynamic_similarity() -> None:
    assert scaled_cadence(Gait.WALK, 1.3, HUMAN_HEIGHT) == pytest.approx(
        human_cadence(Gait.WALK, 1.3)
    )
    # A quarter-height robot steps twice as fast at the speed with the same Froude number.
    quarter = HUMAN_HEIGHT / 4
    assert human_equivalent_speed(0.65, quarter) == pytest.approx(1.3)
    assert scaled_cadence(Gait.WALK, 0.65, quarter) == pytest.approx(
        2 * human_cadence(Gait.WALK, 1.3)
    )
