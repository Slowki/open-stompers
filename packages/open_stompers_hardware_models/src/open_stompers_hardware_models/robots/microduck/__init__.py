"""Pollen Robotics Microduck LinkForge models."""

from collections.abc import Callable
from typing import Final

from linkforge.core import Robot

from open_stompers_hardware_models.mjcf_metadata import MjcfMetadata
from open_stompers_hardware_models.robots.microduck.assets import ensure_meshes
from open_stompers_hardware_models.robots.microduck.model import build
from open_stompers_hardware_models.robots.microduck.sensors import SENSORS, SITES
from open_stompers_hardware_models.robots.microduck.variants import MicroduckVariant

VARIANTS: Final[tuple[MicroduckVariant, ...]] = tuple(MicroduckVariant)
MODELS: Final[dict[MicroduckVariant, Callable[[], Robot]]] = {
    variant: (lambda variant=variant: build(variant)) for variant in VARIANTS
}
MJCF_METADATA_BY_VARIANT: Final[dict[MicroduckVariant, MjcfMetadata]] = {
    MicroduckVariant.LEGGED: MjcfMetadata(
        root_pos=(0.0, 0.0, 0.12),
        root_freejoint="trunk_base_freejoint",
        sites=SITES,
        sensors=SENSORS,
    ),
    MicroduckVariant.ROLLER_SKATES: MjcfMetadata(
        root_pos=(0.0, 0.0, 0.12),
        root_freejoint="trunk_base_freejoint",
    ),
}

__all__ = [
    "MJCF_METADATA_BY_VARIANT",
    "MODELS",
    "VARIANTS",
    "MicroduckVariant",
    "build",
    "ensure_meshes",
]
