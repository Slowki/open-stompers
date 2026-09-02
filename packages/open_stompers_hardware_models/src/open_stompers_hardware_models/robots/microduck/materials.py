"""Microduck material definitions."""

from typing import Final

from open_stompers_hardware_models.materials import PbrMaterialSpec


def _aliases(spec: PbrMaterialSpec, names: tuple[str, ...]) -> dict[str, PbrMaterialSpec]:
    """Map Microduck material names to one physical material specification."""
    return dict.fromkeys(names, spec)


_CYAN_PLASTIC: Final = PbrMaterialSpec((0.45, 0.85, 0.92, 1.0), metallic=0.0, roughness=0.3)
_ORANGE_PLASTIC: Final = PbrMaterialSpec((0.95, 0.22, 0.04, 1.0), metallic=0.0, roughness=0.3)
_YELLOW_PLASTIC: Final = PbrMaterialSpec((0.98, 0.58, 0.02, 1.0), metallic=0.0, roughness=0.3)
_WARM_WHITE_PLASTIC: Final = PbrMaterialSpec((0.82, 0.83, 0.80, 1.0), metallic=0.0, roughness=0.32)
_CHARCOAL_ABS: Final = PbrMaterialSpec((0.035, 0.045, 0.055, 1.0), metallic=0.0, roughness=0.38)
_BLACK_RUBBER: Final = PbrMaterialSpec((0.018, 0.018, 0.016, 1.0), metallic=0.0, roughness=0.72)
_BLACK_LENS: Final = PbrMaterialSpec((0.003, 0.006, 0.01, 1.0), metallic=0.0, roughness=0.08)
_STEEL: Final = PbrMaterialSpec((0.42, 0.45, 0.49, 1.0), metallic=0.98, roughness=0.28)
_PCB: Final = PbrMaterialSpec((0.03, 0.18, 0.08, 1.0), metallic=0.0, roughness=0.5)
_SOFT_LAVENDER: Final = PbrMaterialSpec((0.8, 0.68, 0.8, 1.0), metallic=0.0, roughness=0.55)

PBR_MATERIALS: Final[dict[str, PbrMaterialSpec]] = {
    **_aliases(
        _CYAN_PLASTIC,
        (
            "left_shell_material",
            "right_shell_material",
            "leg_material",
            "top_head_shell_material",
            "trunk_base_material",
            "upper_leg_rigidity_plate_material",
            "yaw_roll_motion_material",
        ),
    ),
    **_aliases(
        _ORANGE_PLASTIC,
        ("bottom_head_shell_material", "jaw_material", "rim_material"),
    ),
    **_aliases(
        _YELLOW_PLASTIC,
        (
            "ankle_l_v1_material",
            "ankle_left_material",
            "ankle_r_v1_material",
            "ankle_right_material",
            "foot_left_material",
            "foot_right_material",
            "sole_left_material",
            "sole_right_material",
        ),
    ),
    **_aliases(
        _WARM_WHITE_PLASTIC,
        ("hip_l_material", "upper_leg_left_material", "upper_leg_right_material"),
    ),
    **_aliases(
        _STEEL,
        (
            "banana_pcb_locker_material",
            "bearing_roll_material",
            "np_f970_material",
            "power_support_material",
            "roller_blade_material",
            "seeed_bearing__configuration__22x16x4_material",
            "seeed_bearing__configuration_default_material",
            "yaw2roll_material",
        ),
    ),
    **_aliases(
        _CHARCOAL_ABS,
        (
            "face_part_material",
            "motor_support_material",
            "neck_material",
            "neck_pitch_material",
            "noenoeil_material",
            "speaker_material",
            "xl330_material",
        ),
    ),
    **_aliases(_BLACK_RUBBER, ("tire_material",)),
    **_aliases(_BLACK_LENS, ("lens_material", "m12_lens_holder_material")),
    **_aliases(
        _PCB,
        ("elec_rpi_robot_hat_pcb_material", "pcb__raspberry_pi_zero_2_w_material"),
    ),
    **_aliases(_SOFT_LAVENDER, ("jaw_soft_material", "soft_mouth_top_material")),
}

MATERIALS: Final[dict[str, tuple[float, float, float, float]]] = {
    name: spec.base_color for name, spec in PBR_MATERIALS.items()
}


__all__ = ["MATERIALS", "PBR_MATERIALS"]
