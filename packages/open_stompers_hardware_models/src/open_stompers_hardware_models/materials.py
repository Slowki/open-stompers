"""Physically based material specifications for hardware model exports."""

import math
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class PbrMaterialSpec:
    """Physically based surface parameters."""

    base_color: tuple[float, float, float, float]
    metallic: float
    roughness: float

    def __post_init__(self) -> None:
        """Validate color and PBR factors."""
        if any(
            not math.isfinite(component) or not 0.0 <= component <= 1.0
            for component in self.base_color
        ):
            raise ValueError("PBR base color components must be finite values in [0.0, 1.0]")
        if not 0.0 <= self.metallic <= 1.0:
            raise ValueError("PBR metallic factor must be in [0.0, 1.0]")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("PBR roughness factor must be in [0.0, 1.0]")


DEFAULT_PBR_MATERIAL: Final = PbrMaterialSpec(
    (0.752941176470588, 0.752941176470588, 0.752941176470588, 1.0),
    metallic=0.0,
    roughness=0.5,
)

__all__ = ["DEFAULT_PBR_MATERIAL", "PbrMaterialSpec"]
