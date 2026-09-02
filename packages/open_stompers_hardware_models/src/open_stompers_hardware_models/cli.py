"""Command line interface for open_stompers_hardware_models."""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from open_stompers_hardware_models import (
    MODELS,
    VARIANTS,
    HardwareVariant,
    MicroduckVariant,
    MiniPiPlusVariant,
    mjcf_metadata_for,
)
from open_stompers_hardware_models.assets import AssetError
from open_stompers_hardware_models.export import ExportFormat, export_model, parse_export_format
from open_stompers_hardware_models.gltf import GltfExportError
from open_stompers_hardware_models.link_transforms import (
    LinkTransformCodegenError,
    check_link_transforms,
    generate_link_transforms,
)
from open_stompers_hardware_models.robots import microduck, mini_pi_plus

_VARIANT_BY_VALUE: dict[str, HardwareVariant] = {variant.value: variant for variant in VARIANTS}


def _cmd_list(_args: argparse.Namespace) -> int:
    for variant in VARIANTS:
        print(variant)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    variant_values = args.variant or [variant.value for variant in VARIANTS]
    if unknown := [value for value in variant_values if value not in _VARIANT_BY_VALUE]:
        print(
            f"error: unknown variant(s): {', '.join(unknown)} (available: {', '.join(VARIANTS)})",
            file=sys.stderr,
        )
        return 2
    for value in variant_values:
        variant = _VARIANT_BY_VALUE[value]
        match variant:
            case MiniPiPlusVariant():
                paths = mini_pi_plus.ensure_meshes(variant)
            case MicroduckVariant():
                paths = microduck.ensure_meshes(variant)
        print(f"{variant}: {len(paths)} meshes ready under the asset cache")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    variant = _VARIANT_BY_VALUE[args.variant]
    robot = MODELS[variant]()
    out = Path(args.output)
    export_model(
        robot,
        out,
        args.format,
        mjcf_metadata=mjcf_metadata_for(variant),
        simplify_ratio=args.simplify_ratio,
    )
    print(f"Wrote {out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open_stompers_hardware_models",
        description=(
            "Inspect, fetch assets for, export, and generate transforms for hardware models."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available hardware models").set_defaults(
        func=_cmd_list
    )

    fetch = subparsers.add_parser("fetch", help="Download mesh assets into the cache")
    fetch.add_argument("variant", nargs="*", help="Variants to fetch (default: all)")
    fetch.set_defaults(func=_cmd_fetch)

    export = subparsers.add_parser("export", help="Export a model")
    export.add_argument("variant", choices=VARIANTS, help="Model variant to export")
    export.add_argument("output", help="Destination file path")
    export.add_argument(
        "--format",
        type=parse_export_format,
        choices=tuple(ExportFormat),
        metavar="{blender,blend,glb,urdf,mjcf,mcjf}",
        help="Output format (default: infer from destination extension, then URDF)",
    )
    export.add_argument(
        "--simplify-ratio",
        type=_parse_simplify_ratio,
        nargs="?",
        const=0.25,
        default=None,
        metavar="KEEP_FRACTION",
        help="Simplify STL meshes, keeping this face fraction (default when enabled: 0.25)",
    )
    export.set_defaults(func=_cmd_export)

    generate = subparsers.add_parser(
        "generate-transforms",
        help="Generate nalgebra link transforms for the hardware adapters",
    )
    generate.add_argument(
        "--output-root",
        type=Path,
        help="Workspace root to write into (default: this repository)",
    )
    generate.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if generated sources differ from the workspace",
    )
    generate.set_defaults(func=_cmd_generate_transforms)

    return parser


def _parse_simplify_ratio(value: str) -> float:
    """Parse and validate a face-retention ratio for simplified GLB export."""
    try:
        ratio = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number in (0, 1]") from error
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return ratio


def _cmd_generate_transforms(args: argparse.Namespace) -> int:
    output_root = args.output_root
    if args.check:
        if mismatches := check_link_transforms(output_root):
            print("error: generated link transforms are out of date:", file=sys.stderr)
            for mismatch in mismatches:
                print(f"error: {mismatch}", file=sys.stderr)
            print(
                "error: run `uv run open_stompers_hardware_models generate-transforms`",
                file=sys.stderr,
            )
            return 1
        print("generated link transforms are up to date")
        return 0
    for path in generate_link_transforms(output_root):
        print(f"Wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``open_stompers_hardware_models`` console script."""
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (AssetError, GltfExportError, LinkTransformCodegenError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
