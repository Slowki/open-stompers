"""Tests for animation storage and conversion."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from open_stompers_animations.conversion import to_motion_clip
from open_stompers_animations.format import METADATA_KEY, read_animation, write_animation
from open_stompers_animations.types import Animation, LoopOffset


def test_round_trip_preserves_everything(duck_walk: Animation, tmp_path: Path) -> None:
    path = write_animation(duck_walk, tmp_path / "walk.parquet")
    restored = read_animation(path)
    assert restored.name == duck_walk.name
    assert restored.kind is duck_walk.kind
    assert restored.joint_names == duck_walk.joint_names
    assert restored.contact_names == duck_walk.contact_names
    assert restored.behaviors == duck_walk.behaviors
    assert restored.loop_offset == pytest.approx(duck_walk.loop_offset)
    assert restored.parameters == duck_walk.parameters
    for attribute in (
        "root_positions",
        "root_quaternions",
        "joint_positions",
        "commands",
        "phases",
        "contacts",
        "center_of_mass",
        "zero_moment_point",
    ):
        np.testing.assert_array_equal(getattr(restored, attribute), getattr(duck_walk, attribute))
    assert METADATA_KEY in (pq.read_schema(path).metadata or {})


def test_rejects_foreign_parquet(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), path)
    with pytest.raises(ValueError, match="not an Open Stompers animation"):
        read_animation(path)


def test_frames_are_fifty_hertz_with_one_row_each(duck_walk: Animation, tmp_path: Path) -> None:
    table = pq.read_table(write_animation(duck_walk, tmp_path / "walk.parquet"))
    times = table.column("time").to_numpy()
    assert len(times) == duck_walk.frame_count
    np.testing.assert_allclose(np.diff(times), 0.02, atol=1e-6)


def test_motion_clip_conversion(pi_walk: Animation) -> None:
    clip = to_motion_clip(pi_walk)
    assert clip.frame_count == pi_walk.frame_count
    commands = clip.metadata["commands"]
    assert isinstance(commands, list)
    assert commands[0] == pytest.approx([0.25, 0.0, 0.0])


def test_loop_offset_algebra() -> None:
    # A non-zero yaw exercises the same rotation lines as the other sample headings.
    offset = LoopOffset(0.3, -0.1, -1.3)
    identity = offset.compose(offset.inverse())
    assert (identity.x, identity.y, identity.yaw) == pytest.approx((0, 0, 0), abs=1e-12)
    twice = offset.power(2)
    point = np.array([[0.2, 0.5, 0.1]])
    np.testing.assert_allclose(
        twice.apply_positions(point), offset.apply_positions(offset.apply_positions(point))
    )
    np.testing.assert_allclose(
        offset.power(-1).apply_positions(offset.apply_positions(point)), point
    )
