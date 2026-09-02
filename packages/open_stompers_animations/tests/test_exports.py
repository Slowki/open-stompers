"""Tests for the command line, Rerun viewer, and Blender export."""

import shutil
import sys
from pathlib import Path

import pytest

from open_stompers_animations import cli
from open_stompers_animations.blender import export_blender_animation
from open_stompers_animations.format import read_animation, write_animation
from open_stompers_animations.types import Animation
from open_stompers_animations.viewer import view_animations


def test_command_line_generates_and_analyzes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "walk.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "animations",
            "generate",
            "walk",
            "--robot",
            "microduck",
            "--velocity",
            "0.05",
            "--output",
            str(output),
            "--analyze",
        ],
    )
    cli.main()
    animation = read_animation(output)
    assert animation.commands[0, 0] == pytest.approx(0.05)
    assert (tmp_path / "walk.analysis.json").exists()


def test_command_line_rejects_foreign_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "animations",
            "generate",
            "sit_down",
            "--robot",
            "microduck",
            "--velocity",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_viewer_writes_a_recording(duck_walk: Animation, tmp_path: Path) -> None:
    path = write_animation(duck_walk, tmp_path / "walk.parquet")
    recording = tmp_path / "walk.rrd"
    view_animations([path], cycles=2, save=recording)
    assert recording.stat().st_size > 0


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")
def test_blender_export(duck_walk: Animation, tmp_path: Path) -> None:
    output = export_blender_animation(duck_walk, tmp_path / "walk.blend")
    assert output.stat().st_size > 0
