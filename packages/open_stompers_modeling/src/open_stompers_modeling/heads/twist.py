"""Heading-frame planar twist shared by command and state outputs."""

from typing import Final

#: Component order of a heading-frame velocity twist.
TWIST_COMPONENTS: Final = ("forward_velocity", "lateral_velocity", "yaw_rate")
#: Width of `TWIST_COMPONENTS`.
TWIST_SIZE: Final = len(TWIST_COMPONENTS)
