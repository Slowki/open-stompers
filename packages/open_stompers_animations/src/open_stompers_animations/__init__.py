"""Procedural loops and actions for Open Stompers robots.

Animations are planned with a variable-height inverted pendulum and zero moment
point preview, solved with whole-body inverse kinematics in MuJoCo, sampled at
50 Hz with velocity command labels, and stored as Parquet.

Example:
    >>> from open_stompers_animations import write_animation
    >>> from open_stompers_animations.generators import GENERATORS
    >>> from open_stompers_hardware_models import MicroduckVariant
    >>> animation = GENERATORS["walk"](MicroduckVariant.LEGGED, 0.08)
    >>> write_animation(animation, "walk.parquet")
"""

from .format import read_animation, write_animation
from .sampling import resample, sample_animation
from .types import Animation, AnimationKind, Behavior, Gait, LoopOffset

__all__ = [
    "Animation",
    "AnimationKind",
    "Behavior",
    "Gait",
    "LoopOffset",
    "read_animation",
    "resample",
    "sample_animation",
    "write_animation",
]
