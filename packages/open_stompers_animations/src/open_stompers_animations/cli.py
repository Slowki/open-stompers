"""Generate, analyze, view, and export robot animations."""

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Final

from .analysis.metrics import analysis_paths, analyze_animation
from .blender import export_blender_animation
from .format import read_animation, write_animation
from .generators import GENERATORS
from .library import generate_library
from .robots import ROBOTS, parse_variant
from .viewer import view_animations

#: Generator keyword arguments settable from the command line.
_GENERATOR_OPTIONS: Final = (
    "velocity",
    "lateral_velocity",
    "yaw_rate",
    "cadence",
    "step_height",
    "walk_velocity",
    "run_velocity",
    "duration",
    "height",
)


def _generate(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Run one generator and store its animation."""
    generator = GENERATORS[arguments.generator]
    accepted = inspect.signature(generator).parameters
    options = {
        name: value
        for name in _GENERATOR_OPTIONS
        if (value := getattr(arguments, name)) is not None
    }
    if unsupported := [name for name in options if name not in accepted]:
        parser.error(
            f"{arguments.generator} does not accept: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
        )
    try:
        animation = generator(parse_variant(arguments.robot), **options)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    output = arguments.output
    if output.suffix != ".parquet":
        output = output / f"{animation.name}.parquet"
    path = write_animation(animation, output)
    print(f"Wrote {path} ({animation.frame_count} frames)")
    if arguments.analyze:
        _report(path)


def _report(path: Path) -> None:
    """Analyze a stored animation and print its key metrics."""
    metrics, _ = analyze_animation(read_animation(path), path)
    summary_path, _ = analysis_paths(path)
    summary = {
        key: value
        for key, value in metrics.to_dict().items()
        if key not in {"joints", "center_of_mass"}
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


def main() -> None:
    """Parse the command line and run the requested command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List generators and supported robots")
    generate = commands.add_parser("generate", help="Generate one animation")
    generate.add_argument("generator", choices=sorted(GENERATORS))
    generate.add_argument("--robot", required=True, help="Hardware variant, e.g. microduck")
    generate.add_argument("--output", type=Path, required=True, help="File or directory")
    generate.add_argument("--analyze", action="store_true", help="Run inverse dynamics")
    for name in _GENERATOR_OPTIONS:
        generate.add_argument(f"--{name.replace('_', '-')}", type=float, dest=name)
    library = commands.add_parser("library", help="Generate the standard animation set")
    library.add_argument("--robot", required=True)
    library.add_argument("--output", type=Path, required=True)
    library.add_argument("--no-analysis", action="store_true")
    analyze = commands.add_parser("analyze", help="Run inverse dynamics on animation files")
    analyze.add_argument("files", nargs="+", type=Path)
    view = commands.add_parser("view", help="Open animation files in Rerun")
    view.add_argument("files", nargs="+", type=Path)
    view.add_argument("--cycles", type=int, default=3, help="Loop repetitions to play")
    view.add_argument("--save", type=Path, help="Write an .rrd recording instead")
    blender = commands.add_parser("export-blender", help="Write a keyframed Blender project")
    blender.add_argument("file", type=Path)
    blender.add_argument("output", type=Path)
    blender.add_argument("--rig", type=Path, help="Reuse an existing rigged .blend")
    arguments = parser.parse_args()
    match arguments.command:
        case "list":
            print("Generators:", ", ".join(GENERATORS))
            print("Robots:", ", ".join(variant.value for variant in ROBOTS))
        case "generate":
            _generate(arguments, parser)
        case "library":
            manifest = generate_library(
                parse_variant(arguments.robot), arguments.output, analyze=not arguments.no_analysis
            )
            print(f"Wrote {manifest}")
        case "analyze":
            for path in arguments.files:
                _report(path)
        case "view":
            view_animations(arguments.files, cycles=arguments.cycles, save=arguments.save)
        case "export-blender":
            path = export_blender_animation(
                read_animation(arguments.file), arguments.output, rig=arguments.rig
            )
            print(f"Wrote {path}")


__all__ = ["main"]
