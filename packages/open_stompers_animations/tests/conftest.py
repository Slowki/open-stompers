"""Shared robots and animations for the animation tests."""

from typing import Final

import pytest

from open_stompers_animations.generators import loops
from open_stompers_animations.types import Animation
from open_stompers_hardware_models import MicroduckVariant, MiniPiPlusVariant

MINI_PI: Final = MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITH_WAIST
MICRODUCK: Final = MicroduckVariant.LEGGED


@pytest.fixture(scope="session")
def duck_walk() -> Animation:
    return loops.walk(MICRODUCK, 0.08)


@pytest.fixture(scope="session")
def pi_walk() -> Animation:
    return loops.walk(MINI_PI, 0.25)
