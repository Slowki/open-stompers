"""Shared fixtures for open_stompers_hardware_models tests."""

from pathlib import Path

import pytest
from linkforge.core import Robot, read_urdf

from open_stompers_hardware_models.robots import mini_pi_plus
from open_stompers_hardware_models.robots.mini_pi_plus import MiniPiPlusVariant

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def upstream_urdf_path(variant: MiniPiPlusVariant) -> Path:
    return FIXTURES_DIR / f"{variant}.urdf"


@pytest.fixture(params=mini_pi_plus.VARIANTS)
def variant(request: pytest.FixtureRequest) -> MiniPiPlusVariant:
    return request.param


@pytest.fixture
def upstream_robot(variant: MiniPiPlusVariant) -> Robot:
    """Return the upstream URDF for the variant, parsed by LinkForge."""
    return read_urdf(upstream_urdf_path(variant))
