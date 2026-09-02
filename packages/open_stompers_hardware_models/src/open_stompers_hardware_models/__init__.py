"""Hardware models for Open Stompers.

Binary mesh assets are downloaded on demand into `$XDG_CACHE_HOME/openstompers`.

Example:
    >>> from open_stompers_hardware_models import MiniPiPlusVariant, load
    >>> from linkforge.core import URDFGenerator
    >>> robot = load(MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITH_WAIST)
    >>> urdf_xml = URDFGenerator().generate(robot)
"""

from collections.abc import Callable
from typing import Final

from linkforge.core import Robot

from open_stompers_hardware_models import assets
from open_stompers_hardware_models.materials import PbrMaterialSpec
from open_stompers_hardware_models.mjcf_metadata import MjcfMetadata
from open_stompers_hardware_models.robots import microduck, mini_pi_plus
from open_stompers_hardware_models.variants import (
    HardwareVariant,
    MicroduckVariant,
    MiniPiPlusVariant,
)

MODELS: Final[dict[HardwareVariant, Callable[[], Robot]]] = {
    **mini_pi_plus.MODELS,
    **microduck.MODELS,
}

VARIANTS: Final[tuple[HardwareVariant, ...]] = tuple(MODELS)


def load(variant: HardwareVariant) -> Robot:
    """Build the validated model for a hardware variant.

    Args:
        variant: Hardware variant to build.

    Returns:
        The validated LinkForge robot.

    Raises:
        KeyError: If the hardware variant is unknown.
    """
    try:
        builder = MODELS[variant]
    except KeyError:
        available = ", ".join(VARIANTS)
        raise KeyError(f"Unknown hardware model '{variant}'. Available: {available}") from None
    return builder()


def mjcf_metadata_for(variant: HardwareVariant) -> MjcfMetadata:
    """Return the MJCF export metadata for a hardware variant."""
    return microduck.MJCF_METADATA_BY_VARIANT.get(variant, MjcfMetadata())


__all__ = [
    "MODELS",
    "VARIANTS",
    "HardwareVariant",
    "MicroduckVariant",
    "MiniPiPlusVariant",
    "MjcfMetadata",
    "PbrMaterialSpec",
    "assets",
    "load",
    "microduck",
    "mini_pi_plus",
    "mjcf_metadata_for",
]
