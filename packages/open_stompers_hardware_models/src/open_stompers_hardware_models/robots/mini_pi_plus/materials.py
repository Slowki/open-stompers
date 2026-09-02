"""Mini Pi Plus material definitions."""

from typing import Final

from open_stompers_hardware_models.materials import PbrMaterialSpec

MATERIAL_NAME: Final[str] = "gray"
ALUMINIUM_MATERIAL_NAME: Final[str] = "aluminium"
BLACK_ABS_MATERIAL_NAME: Final[str] = "black_abs"

_GRAY: Final = PbrMaterialSpec(
    (0.752941176470588, 0.752941176470588, 0.752941176470588, 1.0),
    metallic=0.0,
    roughness=0.5,
)
_ALUMINIUM: Final = PbrMaterialSpec(
    (0.92, 0.94, 0.97, 1.0),
    metallic=0.92,
    roughness=0.24,
)
_BLACK_ABS: Final = PbrMaterialSpec(
    (0.012, 0.015, 0.018, 1.0),
    metallic=0.0,
    roughness=0.34,
)

PBR_MATERIALS: Final[dict[str, PbrMaterialSpec]] = {
    MATERIAL_NAME: _GRAY,
    ALUMINIUM_MATERIAL_NAME: _ALUMINIUM,
    BLACK_ABS_MATERIAL_NAME: _BLACK_ABS,
}

MATERIALS: Final[dict[str, tuple[float, float, float, float]]] = {
    name: spec.base_color for name, spec in PBR_MATERIALS.items()
}

GRAY_RGBA: Final[tuple[float, float, float, float]] = MATERIALS[MATERIAL_NAME]
ALUMINIUM_RGBA: Final[tuple[float, float, float, float]] = MATERIALS[ALUMINIUM_MATERIAL_NAME]
BLACK_ABS_RGBA: Final[tuple[float, float, float, float]] = MATERIALS[BLACK_ABS_MATERIAL_NAME]


__all__ = [
    "ALUMINIUM_MATERIAL_NAME",
    "ALUMINIUM_RGBA",
    "BLACK_ABS_MATERIAL_NAME",
    "BLACK_ABS_RGBA",
    "GRAY_RGBA",
    "MATERIALS",
    "MATERIAL_NAME",
    "PBR_MATERIALS",
]
