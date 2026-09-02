"""Mini Pi Plus robot family (HighTorque Robotics PiPlus S)."""

from collections.abc import Callable
from typing import Final

from linkforge.core import Robot

from open_stompers_hardware_models.robots.mini_pi_plus import (
    piplus_s_12l8a0g2h0w,
    piplus_s_12l8a0g2h1w,
    piplus_s_12l10a0g0h1w,
)
from open_stompers_hardware_models.robots.mini_pi_plus.assets import (
    FAMILY,
    MESH_MANIFESTS,
    UPSTREAM_BRANCH,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    ensure_meshes,
    get_cache_relative_path,
    mesh_entries,
    upstream_url,
)
from open_stompers_hardware_models.robots.mini_pi_plus.variants import MiniPiPlusVariant

MODELS: Final[dict[MiniPiPlusVariant, Callable[[], Robot]]] = {
    piplus_s_12l10a0g0h1w.VARIANT: piplus_s_12l10a0g0h1w.build,
    piplus_s_12l8a0g2h0w.VARIANT: piplus_s_12l8a0g2h0w.build,
    piplus_s_12l8a0g2h1w.VARIANT: piplus_s_12l8a0g2h1w.build,
}

VARIANTS: Final[tuple[MiniPiPlusVariant, ...]] = tuple(MODELS)

__all__ = [
    "FAMILY",
    "MESH_MANIFESTS",
    "MODELS",
    "UPSTREAM_BRANCH",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "VARIANTS",
    "MiniPiPlusVariant",
    "build",
    "ensure_meshes",
    "get_cache_relative_path",
    "mesh_entries",
    "piplus_s_12l8a0g2h0w",
    "piplus_s_12l8a0g2h1w",
    "piplus_s_12l10a0g0h1w",
    "upstream_url",
]


def build(variant: MiniPiPlusVariant) -> Robot:
    """Build the LinkForge model for a Mini Pi Plus variant."""
    try:
        builder = MODELS[variant]
    except KeyError:
        msg = f"Unknown Mini Pi Plus variant '{variant}'. Available: {', '.join(VARIANTS)}"
        raise KeyError(msg) from None
    return builder()
