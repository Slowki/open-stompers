"""Human cadence data scaled to robots by dynamic similarity.

Walking follows the walk ratio, step length divided by cadence, which stays near
0.0064 m per step/min across preferred and faster free walking (Sekiya, Nagasaki,
Ito and Furuna 1996). With `speed = step length * cadence / 60` this gives
`cadence = sqrt(60 * speed / walk ratio)`, about 110 steps/min at 1.3 m/s.

Running cadence rises slowly with speed: runners averaging 1.75 m tall chose 154,
158 and 160 steps/min at 2.4, 2.6 and 2.9 m/s ("Optimal stride frequencies in
running at different speeds", PLOS ONE 12(10): e0184273, 2017), and about 165
steps/min at 3.0 m/s outdoors, rising by about 3 steps/min for 15 % more speed
("Individual optimal step frequency during outdoor running", 2020). A line through
160 steps/min at 2.7 m/s with 9 steps/min per m/s fits both.

Robots are mapped onto that data by equal Froude number `v^2 / (g h)`: for a height
ratio `s = h_robot / h_human`, a robot speed `v` matches the human speed
`v / sqrt(s)`, and the human cadence at that speed is multiplied by `1 / sqrt(s)`.
"""

from math import sqrt
from typing import Final

from ..types import Gait

#: Reference human standing height in meters, the mean of the running study subjects.
HUMAN_HEIGHT: Final = 1.75
#: Human walk ratio in meters per step/min.
WALK_RATIO: Final = 0.0064
#: Slowest human walking speed in meters per second the walk ratio is applied to.
SLOWEST_WALK: Final = 0.5
#: Human running cadence in steps/min at the reference running speed.
RUNNING_CADENCE: Final = 160.0
#: Reference human running speed in meters per second.
RUNNING_SPEED: Final = 2.7
#: Increase of human running cadence in steps/min per meter per second.
RUNNING_CADENCE_SLOPE: Final = 9.0
#: Slowest human running cadence in steps/min; slower runners walk instead.
SLOWEST_RUNNING_CADENCE: Final = 150.0
#: Fastest human running cadence in steps/min used here.
FASTEST_RUNNING_CADENCE: Final = 200.0


def human_cadence(gait: Gait, speed: float) -> float:
    """Return the average human cadence for a gait and speed.

    Args:
        gait: Walking, jogging, or running; standing uses walking.
        speed: Ground speed in meters per second.

    Returns:
        Steps per minute.
    """
    if gait in {Gait.WALK, Gait.STAND}:
        return sqrt(60.0 * max(speed, SLOWEST_WALK) / WALK_RATIO)
    cadence = RUNNING_CADENCE + RUNNING_CADENCE_SLOPE * (speed - RUNNING_SPEED)
    return min(max(cadence, SLOWEST_RUNNING_CADENCE), FASTEST_RUNNING_CADENCE)


def scaled_cadence(gait: Gait, speed: float, height: float) -> float:
    """Return the human cadence at the dynamically similar speed, scaled to a robot.

    Args:
        gait: Walking, jogging, or running.
        speed: Robot ground speed in meters per second.
        height: Robot standing height in meters.

    Returns:
        Robot steps per minute.
    """
    scale = sqrt(height / HUMAN_HEIGHT)
    return human_cadence(gait, speed / scale) / scale


def human_equivalent_speed(speed: float, height: float) -> float:
    """Return the human speed with the same Froude number as a robot speed."""
    return speed / sqrt(height / HUMAN_HEIGHT)


__all__ = [
    "HUMAN_HEIGHT",
    "WALK_RATIO",
    "human_cadence",
    "human_equivalent_speed",
    "scaled_cadence",
]
