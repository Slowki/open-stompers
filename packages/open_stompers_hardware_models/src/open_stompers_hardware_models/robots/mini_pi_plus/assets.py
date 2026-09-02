"""Upstream mesh assets for the Mini Pi Plus family.

Meshes live in the upstream ``HT_Robot_URDF`` repository and are downloaded on
demand into the shared asset cache (see ``open_stompers_hardware_models.assets``). They are
pinned to a specific upstream commit and verified with SHA-256 checksums.
"""

from pathlib import Path
from typing import Final

from open_stompers_hardware_models.assets import AssetEntry, fetch
from open_stompers_hardware_models.robots.mini_pi_plus.manifests import MESH_MANIFESTS
from open_stompers_hardware_models.robots.mini_pi_plus.variants import MiniPiPlusVariant

UPSTREAM_REPO: Final[str] = "HighTorque-Robotics/HT_Robot_URDF"
UPSTREAM_BRANCH: Final[str] = "PiPlus_URDF"
UPSTREAM_COMMIT: Final[str] = "c05257c64c79b624e773eae704752c09c6bdb007"

FAMILY: Final[str] = "mini_pi_plus"

_RAW_BASE: Final[str] = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/ht_urdf"
)


def get_cache_relative_path(variant: MiniPiPlusVariant, filename: str) -> str:
    """Return the cache-root-relative resource path for a mesh file."""
    return f"hardware_models/{FAMILY}/{variant}/meshes/{filename}"


def upstream_url(variant: MiniPiPlusVariant, filename: str) -> str:
    """Return the pinned upstream download URL for a mesh file."""
    return f"{_RAW_BASE}/{variant}/meshes/{filename}"


def mesh_entries(variant: MiniPiPlusVariant) -> tuple[AssetEntry, ...]:
    """Return the downloadable asset entries for one variant."""
    manifest = MESH_MANIFESTS[variant]
    return tuple(
        AssetEntry(
            resource=get_cache_relative_path(variant, filename),
            url=upstream_url(variant, filename),
            sha256=sha,
        )
        for filename, sha in sorted(manifest.items())
    )


def ensure_meshes(variant: MiniPiPlusVariant | None = None) -> list[Path]:
    """Download missing meshes for one or all variants into the cache."""
    variants = [variant] if variant is not None else sorted(MESH_MANIFESTS)
    entries = [entry for current in variants for entry in mesh_entries(current)]
    return fetch(entries)
