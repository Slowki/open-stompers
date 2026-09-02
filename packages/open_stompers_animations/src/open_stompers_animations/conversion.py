"""Conversion of animations into dataset motion clips."""

from open_stompers_datasets.types import MotionClip

from .types import Animation


def to_motion_clip(animation: Animation) -> MotionClip:
    """Convert an animation into a datasets motion clip.

    Args:
        animation: Animation to convert.

    Returns:
        A motion clip carrying the animation labels in its metadata.
    """
    return MotionClip(
        root_positions=animation.root_positions,
        root_quaternions=animation.root_quaternions,
        joint_positions=animation.joint_positions,
        joint_names=animation.joint_names,
        sample_interval=animation.sample_interval,
        metadata={
            "robot": animation.variant,
            "source": "open_stompers_animations",
            "motion_id": animation.name,
            "kind": str(animation.kind),
            "parameters": animation.parameters,
            "commands": animation.commands.tolist(),
            "behaviors": [str(behavior) for behavior in animation.behaviors],
            **animation.metadata,
        },
    )


__all__ = ["to_motion_clip"]
