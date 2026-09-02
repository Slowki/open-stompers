"""Loop and action animation generators."""

from collections.abc import Callable
from typing import Final

from ..types import Animation
from .actions import (
    jump_vertical,
    run_to_stop,
    run_to_walk,
    sit_down,
    stand_up,
    start_walking,
    stop_walking,
    walk_to_run,
)
from .loops import jog, run, side_step, turn_in_place, turn_while_walking, walk

#: Every generator by name; each takes the hardware variant first.
GENERATORS: Final[dict[str, Callable[..., Animation]]] = {
    function.__name__: function
    for function in (
        walk,
        jog,
        run,
        turn_in_place,
        turn_while_walking,
        side_step,
        start_walking,
        stop_walking,
        walk_to_run,
        run_to_walk,
        run_to_stop,
        sit_down,
        stand_up,
        jump_vertical,
    )
}

__all__ = ["GENERATORS"]
