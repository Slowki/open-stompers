"""Pinned upstream mesh assets for the Microduck models."""

from pathlib import Path
from typing import Final

from open_stompers_hardware_models.assets import AssetEntry, fetch
from open_stompers_hardware_models.robots.microduck.manifests import MESH_MANIFESTS
from open_stompers_hardware_models.robots.microduck.variants import MicroduckVariant

UPSTREAM_REPO: Final[str] = "pollen-robotics/microduck_rl"
UPSTREAM_BRANCH: Final[str] = "main"
UPSTREAM_COMMIT: Final[str] = "1e79c29c97d8b38aee9eefde77a545860ba7658e"
FAMILY: Final[str] = "microduck"


def get_cache_relative_path(variant: MicroduckVariant, filename: str) -> str:
    """Return the cache-relative path for a mesh."""
    return f"hardware_models/{FAMILY}/{variant}/meshes/{filename}"


def upstream_url(_variant: MicroduckVariant, filename: str) -> str:
    """Return the pinned upstream URL for a mesh."""
    return (
        f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/"
        f"src/mjlab_microduck/robot/microduck/assets/{filename}"
    )


def mesh_entries(variant: MicroduckVariant) -> tuple[AssetEntry, ...]:
    """Return manifest entries for one Microduck variant."""
    return tuple(
        AssetEntry(
            resource=get_cache_relative_path(variant, filename),
            url=upstream_url(variant, filename),
            sha256=sha,
        )
        for filename, sha in sorted(MESH_MANIFESTS[variant].items())
    )


def ensure_meshes(variant: MicroduckVariant | None = None) -> list[Path]:
    """Download missing meshes for one or all Microduck variants."""
    variants = [variant] if variant is not None else sorted(MESH_MANIFESTS)
    return fetch(entry for current in variants for entry in mesh_entries(current))
