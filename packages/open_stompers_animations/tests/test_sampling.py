"""Tests for time-based animation sampling."""

import numpy as np
import pytest

from open_stompers_animations.sampling import resample, sample_animation
from open_stompers_animations.types import Animation, AnimationKind


def test_sampling_at_frame_times_returns_frames(pi_walk: Animation) -> None:
    sample = sample_animation(pi_walk, pi_walk.times)
    np.testing.assert_allclose(sample.root_positions, pi_walk.root_positions, atol=1e-6)
    np.testing.assert_allclose(sample.joint_positions, pi_walk.joint_positions, atol=1e-6)
    assert sample.behaviors == pi_walk.behaviors


def test_loops_tile_with_the_loop_offset(pi_walk: Animation) -> None:
    first = sample_animation(pi_walk, [0.0])
    next_cycle = sample_animation(pi_walk, [pi_walk.duration])
    expected = pi_walk.loop_offset.apply_positions(first.root_positions.astype(np.float64))
    np.testing.assert_allclose(next_cycle.root_positions, expected, atol=1e-6)
    np.testing.assert_allclose(next_cycle.joint_positions, first.joint_positions, atol=1e-6)


def test_loop_seam_is_as_smooth_as_the_rest(pi_walk: Animation) -> None:
    times = np.arange(3 * pi_walk.frame_count) * pi_walk.sample_interval
    joints = sample_animation(pi_walk, times).joint_positions
    steps = np.abs(np.diff(joints, axis=0)).max(axis=1)
    seam = steps[pi_walk.frame_count - 1]
    assert seam <= 1.5 * np.percentile(steps, 95)


def test_resampling_a_loop_keeps_its_cycle(pi_walk: Animation) -> None:
    doubled = resample(pi_walk, pi_walk.sample_interval / 2)
    assert doubled.kind is AnimationKind.LOOP
    assert doubled.frame_count == 2 * pi_walk.frame_count
    assert doubled.duration == pytest.approx(pi_walk.duration)


def test_resampling_rejects_partial_cycles(pi_walk: Animation) -> None:
    with pytest.raises(ValueError, match="whole number"):
        resample(pi_walk, 0.013)
