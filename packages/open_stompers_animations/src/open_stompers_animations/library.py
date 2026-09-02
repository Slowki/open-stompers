"""Batch generation of a standard animation library."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from open_stompers_hardware_models import HardwareVariant

from .analysis.metrics import analyze_animation
from .format import write_animation
from .generators import GENERATORS
from .model import build_robot_model
from .planning.commands import default_cadence
from .robots import ROBOTS
from .types import Gait

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipSpecification:
    """One library entry: a generator and its keyword arguments."""

    #: Library file stem.
    name: str
    #: Generator name from `GENERATORS`.
    generator: str
    #: Keyword arguments passed after the hardware variant.
    arguments: dict[str, float] = field(default_factory=dict)


def default_library(variant: HardwareVariant) -> list[ClipSpecification]:
    """List the standard loops and actions of a robot.

    Speeds come from the robot's gait defaults and cadences from scaled human data, so the grid
    suits robots of any size.

    Args:
        variant: Hardware variant to describe.

    Returns:
        Clip specifications with unique names.
    """
    robot = ROBOTS[variant]
    walking, jogging, running = (robot.gaits[gait] for gait in (Gait.WALK, Gait.JOG, Gait.RUN))
    height = build_robot_model(variant).height
    # Each speed at its human-like cadence and at a slower and a quicker step rate.
    clips = [
        ClipSpecification(
            f"walk_{speed:.3f}mps_{cadence:.0f}spm",
            "walk",
            {"velocity": speed, "cadence": cadence},
        )
        for speed in (
            0.4 * walking.default_speed,
            0.7 * walking.default_speed,
            walking.default_speed,
            1.2 * walking.default_speed,
        )
        for factor in (0.85, 1.0, 1.15)
        if (cadence := factor * default_cadence(robot, Gait.WALK, speed, height))
    ]
    clips.append(
        ClipSpecification(
            f"walk_backward_{0.5 * walking.default_speed:.3f}mps",
            "walk",
            {"velocity": -0.5 * walking.default_speed},
        )
    )
    clips += [
        ClipSpecification(f"jog_{speed:.3f}mps", "jog", {"velocity": speed})
        for speed in (
            0.8 * jogging.default_speed,
            jogging.default_speed,
            1.15 * jogging.default_speed,
        )
    ]
    clips += [
        ClipSpecification(f"run_{speed:.3f}mps", "run", {"velocity": speed})
        for speed in (0.85 * running.default_speed, running.default_speed)
    ]
    clips += [
        ClipSpecification(f"turn_in_place_{rate:+.2f}radps", "turn_in_place", {"yaw_rate": rate})
        for rate in (-0.6, -0.3, 0.3, 0.6)
    ]
    clips += [
        ClipSpecification(
            f"turn_while_walking_{rate:+.2f}radps", "turn_while_walking", {"yaw_rate": rate}
        )
        for rate in (-0.3, 0.3)
    ]
    clips += [
        ClipSpecification(f"side_step_{speed:+.3f}mps", "side_step", {"lateral_velocity": speed})
        for speed in (-robot.side_step_speed, robot.side_step_speed)
    ]
    loops = {"walk", "jog", "run", "turn_in_place", "turn_while_walking", "side_step"}
    clips += [ClipSpecification(name, name) for name in GENERATORS if name not in loops]
    return clips


def generate_library(
    variant: HardwareVariant,
    output: str | Path,
    *,
    specifications: list[ClipSpecification] | None = None,
    analyze: bool = True,
) -> Path:
    """Generate, store, and optionally analyze a set of animations.

    Clips that exceed the robot reach are skipped and recorded as failed in the
    manifest instead of aborting the batch.

    Args:
        variant: Hardware variant to animate.
        output: Directory receiving the Parquet files and `manifest.json`.
        specifications: Clips to generate, the default library when omitted.
        analyze: Whether to run inverse dynamics on every clip.

    Returns:
        The manifest path.
    """
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for clip in specifications or default_library(variant):
        entry: dict[str, object] = {
            "name": clip.name,
            "generator": clip.generator,
            "arguments": clip.arguments,
        }
        try:
            animation = GENERATORS[clip.generator](variant, **clip.arguments)
        except (ValueError, RuntimeError) as error:
            _LOGGER.warning("Skipping %s: %s", clip.name, error)
            entries.append({**entry, "status": "failed", "error": str(error)})
            continue
        path = write_animation(animation, directory / f"{clip.name}.parquet")
        entry |= {
            "status": "generated",
            "file": path.name,
            "kind": str(animation.kind),
            "frames": animation.frame_count,
            "duration": animation.duration,
            "mean_command": animation.commands.mean(axis=0).astype(float).tolist(),
            "parameters": animation.parameters,
        }
        if analyze:
            metrics, _ = analyze_animation(animation, path)
            entry["metrics"] = {
                key: value
                for key, value in metrics.to_dict().items()
                if key not in {"joints", "name", "variant", "kind"}
            }
        _LOGGER.info("Generated %s", clip.name)
        entries.append(entry)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps({"variant": str(variant), "animations": entries}, indent=2) + "\n"
    )
    return manifest


__all__ = ["ClipSpecification", "default_library", "generate_library"]
