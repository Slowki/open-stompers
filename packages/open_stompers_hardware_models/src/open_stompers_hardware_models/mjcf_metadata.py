"""MuJoCo metadata for LinkForge hardware exports."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MjcfSite:
    """A named measurement frame relative to a hardware link."""

    name: str
    body: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MjcfSensor:
    """A built-in MuJoCo sensor referencing a site or a body's subtree."""

    name: str
    kind: Literal["framequat", "gyro", "velocimeter", "accelerometer", "subtreeangmom"]
    target: str
    noise: float = 0.0


@dataclass(frozen=True, slots=True)
class MjcfMetadata:
    """MuJoCo-specific options for exporting a LinkForge robot."""

    root_pos: tuple[float, float, float] | None = None
    root_freejoint: str | None = None
    sites: tuple[MjcfSite, ...] = ()
    sensors: tuple[MjcfSensor, ...] = ()
