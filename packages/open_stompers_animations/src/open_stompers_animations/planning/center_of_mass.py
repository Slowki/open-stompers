"""Center of mass planning with a variable-height inverted pendulum.

Both planners solve one sparse linear least-squares problem over the whole
horizon, a batch form of Kajita's zero moment point preview control:

* The vertical planner tracks height targets with smooth accelerations and
  enforces free fall (`z'' = -g`) during flight phases.
* The horizontal planner tracks a zero moment point reference using the moment
  balance `(g + z'') (c - p) = z c''` of a point mass at height `z`, which
  reduces to `c'' = 0` during flight without special cases.

Loops use periodic boundaries in which the sample after the last one is the
first sample moved by the loop offset; actions start and end at rest.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..types import LoopOffset

#: Gravitational acceleration in meters per second squared.
GRAVITY: Final = 9.81
#: Vertical load fraction below which a sample is treated as airborne.
FLIGHT_LOAD: Final = 0.05


@dataclass(frozen=True)
class PlannerWeights:
    """Least-squares weights of the center of mass planners."""

    #: Zero moment point tracking.
    zero_moment_point: float = 1.0
    #: Horizontal jerk.
    horizontal_jerk: float = 1e-7
    #: Vertical acceleration while supported.
    vertical_acceleration: float = 1e-3
    #: Vertical jerk.
    vertical_jerk: float = 1e-8


@dataclass(frozen=True)
class CenterOfMassPlan:
    """Planned center of mass trajectory on a uniform grid."""

    #: Positions shaped (samples, 3).
    positions: NDArray[np.float64]
    #: Velocities shaped (samples, 3).
    velocities: NDArray[np.float64]
    #: Accelerations shaped (samples, 3).
    accelerations: NDArray[np.float64]
    #: Resulting zero moment point shaped (samples, 2), the ground projection when airborne.
    zero_moment_point: NDArray[np.float64]


class _LeastSquares:
    """Sparse least-squares assembler over a sampled vector sequence."""

    def __init__(
        self, count: int, dimensions: int, *, periodic: bool, offset: LoopOffset | None = None
    ) -> None:
        self.count, self.dimensions, self.periodic = count, dimensions, periodic
        self.offset = offset or LoopOffset()
        self._transforms: dict[int, tuple[NDArray, NDArray]] = {}
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.targets: list[float] = []
        self._constraints = _LeastSquaresRows()

    def _transform(self, cycle: int) -> tuple[NDArray, NDArray]:
        """Return the rotation and translation that map cycle zero onto `cycle`."""
        if cycle not in self._transforms:
            if self.dimensions == 1:
                self._transforms[cycle] = (np.eye(1), np.zeros(1))
            else:
                power = self.offset.power(cycle)
                cosine, sine = np.cos(power.yaw), np.sin(power.yaw)
                self._transforms[cycle] = (
                    np.array([[cosine, -sine], [sine, cosine]]),
                    np.array([power.x, power.y]),
                )
        return self._transforms[cycle]

    def add(
        self,
        center: int,
        stencil: list[tuple[int, float]],
        target: NDArray[np.float64] | float,
        weight: float,
    ) -> None:
        """Add the cost rows `sum(coefficient * x[center + shift]) = target`, one per dimension."""
        if weight > 0:
            self._append(self, center, stencil, target, np.sqrt(weight))

    def constrain(
        self, center: int, stencil: list[tuple[int, float]], target: NDArray[np.float64] | float
    ) -> None:
        """Require `sum(coefficient * x[center + shift]) = target` exactly."""
        self._append(self._constraints, center, stencil, target, 1.0)

    def _append(
        self,
        destination: "_LeastSquaresRows | _LeastSquares",
        center: int,
        stencil: list[tuple[int, float]],
        target: NDArray[np.float64] | float,
        scale: float,
    ) -> None:
        """Append scaled rows, resolving periodic indices through the loop transform."""
        indices = [center + shift for shift, _ in stencil]
        if not self.periodic and (min(indices) < 0 or max(indices) >= self.count):
            return
        goal = np.broadcast_to(np.asarray(target, dtype=np.float64), (self.dimensions,))
        for dimension in range(self.dimensions):
            row = len(destination.targets)
            constant = 0.0
            for (_, coefficient), index in zip(stencil, indices, strict=True):
                cycle, base = divmod(index, self.count)
                rotation, translation = self._transform(cycle)
                for component in range(self.dimensions):
                    destination.rows.append(row)
                    destination.columns.append(base * self.dimensions + component)
                    destination.values.append(scale * coefficient * rotation[dimension, component])
                constant += coefficient * translation[dimension]
            destination.targets.append(scale * (goal[dimension] - constant))

    def solve(self) -> NDArray[np.float64]:
        """Solve the equality-constrained problem and return values (count, dimensions)."""
        size = self.count * self.dimensions
        matrix = sparse.csr_matrix(
            (self.values, (self.rows, self.columns)), shape=(len(self.targets), size)
        )
        normal = (matrix.T @ matrix).tocsc() + sparse.identity(size, format="csc") * 1e-12
        right = matrix.T @ np.asarray(self.targets)
        if constraints := len(self._constraints.targets):
            equality = sparse.csr_matrix(
                (self._constraints.values, (self._constraints.rows, self._constraints.columns)),
                shape=(constraints, size),
            )
            normal = sparse.bmat([[normal, equality.T], [equality, None]], format="csc")
            right = np.concatenate((right, self._constraints.targets))
        solution = np.asarray(spsolve(normal, right))[:size]
        return solution.reshape(self.count, self.dimensions)


class _LeastSquaresRows:
    """Sparse row storage."""

    def __init__(self) -> None:
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.targets: list[float] = []


def _stencils(interval: float) -> dict[str, list[tuple[int, float]]]:
    """Return finite-difference stencils for a sample interval."""
    return {
        "velocity_start": [(0, -1 / interval), (1, 1 / interval)],
        "velocity_end": [(-1, -1 / interval), (0, 1 / interval)],
        "acceleration": [(-1, 1 / interval**2), (0, -2 / interval**2), (1, 1 / interval**2)],
        "acceleration_start": [(0, 1 / interval**2), (1, -2 / interval**2), (2, 1 / interval**2)],
        "acceleration_end": [(-2, 1 / interval**2), (-1, -2 / interval**2), (0, 1 / interval**2)],
        "jerk": [
            (-1, -1 / interval**3),
            (0, 3 / interval**3),
            (1, -3 / interval**3),
            (2, 1 / interval**3),
        ],
    }


def _add_rest_boundaries(problem: _LeastSquares, interval: float) -> None:
    """Pin velocity and acceleration to zero at both ends."""
    stencils = _stencils(interval)
    last = problem.count - 1
    for center, names in (
        (0, ("velocity_start", "acceleration_start")),
        (last, ("velocity_end", "acceleration_end")),
    ):
        for name in names:
            problem.constrain(center, stencils[name], 0.0)


def _derivatives(
    values: NDArray[np.float64], interval: float, *, periodic: bool, offset: LoopOffset
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute central-difference velocities and accelerations of samples (count, dims)."""
    if periodic:
        if values.shape[1] == 2:
            before = offset.inverse().apply_positions(
                np.concatenate((values[-1:], np.zeros((1, 1))), axis=1)
            )[:, :2]
            after = offset.apply_positions(np.concatenate((values[:1], np.zeros((1, 1))), axis=1))[
                :, :2
            ]
        else:
            before, after = values[-1:], values[:1]
        padded = np.concatenate((before, values, after))
    else:
        padded = np.concatenate(
            (2 * values[:1] - values[1:2], values, 2 * values[-1:] - values[-2:-1])
        )
    velocities = (padded[2:] - padded[:-2]) / (2 * interval)
    accelerations = (padded[2:] - 2 * padded[1:-1] + padded[:-2]) / interval**2
    if not periodic:
        accelerations[[0, -1]] = 0.0
    return velocities, accelerations


def plan_vertical(
    interval: float,
    heights: NDArray[np.float64],
    height_weights: NDArray[np.float64],
    airborne: NDArray[np.bool_],
    *,
    periodic: bool,
    weights: PlannerWeights | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Plan the center of mass height.

    Args:
        interval: Grid spacing in seconds.
        heights: Height targets shaped (samples,).
        height_weights: Weight of each height target shaped (samples,).
        airborne: Samples in free fall shaped (samples,).
        periodic: Whether the sequence repeats.
        weights: Planner weights, defaults when omitted.

    Returns:
        Heights, vertical velocities, and vertical accelerations, each (samples,).
    """
    weights = weights or PlannerWeights()
    count = len(heights)
    problem = _LeastSquares(count, 1, periodic=periodic)
    stencils = _stencils(interval)
    for index in range(count):
        problem.add(index, stencils["jerk"], 0.0, weights.vertical_jerk)
        if airborne[index]:
            problem.constrain(index, stencils["acceleration"], -GRAVITY)
        else:
            problem.add(index, stencils["acceleration"], 0.0, weights.vertical_acceleration)
        problem.add(index, [(0, 1.0)], heights[index], float(height_weights[index]))
    if not periodic:
        _add_rest_boundaries(problem, interval)
    values = problem.solve()
    velocities, accelerations = _derivatives(
        values, interval, periodic=periodic, offset=LoopOffset()
    )
    if not periodic:
        accelerations[[0, -1]] = 0.0
    return values[:, 0], velocities[:, 0], accelerations[:, 0]


def plan_center_of_mass(
    interval: float,
    reference: NDArray[np.float64],
    heights: NDArray[np.float64],
    vertical_velocities: NDArray[np.float64],
    vertical_accelerations: NDArray[np.float64],
    *,
    periodic: bool,
    offset: LoopOffset | None = None,
    weights: PlannerWeights | None = None,
) -> CenterOfMassPlan:
    """Plan the horizontal center of mass for a zero moment point reference.

    Args:
        interval: Grid spacing in seconds.
        reference: Zero moment point reference shaped (samples, 2).
        heights: Planned center of mass heights above the ground shaped (samples,).
        vertical_velocities: Planned vertical velocities shaped (samples,).
        vertical_accelerations: Planned vertical accelerations shaped (samples,).
        periodic: Whether the sequence repeats.
        offset: Transform from one cycle to the next for periodic plans.
        weights: Planner weights, defaults when omitted.

    Returns:
        The full center of mass plan.
    """
    weights = weights or PlannerWeights()
    offset = offset or LoopOffset()
    count = len(reference)
    problem = _LeastSquares(count, 2, periodic=periodic, offset=offset)
    stencils = _stencils(interval)
    load = (GRAVITY + vertical_accelerations) / GRAVITY
    for index in range(count):
        problem.add(index, stencils["jerk"], np.zeros(2), weights.horizontal_jerk)
        if load[index] < FLIGHT_LOAD:
            problem.constrain(index, stencils["acceleration"], np.zeros(2))
            continue
        lever = heights[index] / GRAVITY
        stencil = [
            (-1, -lever / interval**2),
            (0, load[index] + 2 * lever / interval**2),
            (1, -lever / interval**2),
        ]
        problem.add(index, stencil, load[index] * reference[index], weights.zero_moment_point)
    if not periodic:
        _add_rest_boundaries(problem, interval)
    planar = problem.solve()
    velocities, accelerations = _derivatives(planar, interval, periodic=periodic, offset=offset)
    supported = load >= FLIGHT_LOAD
    zero_moment_point = planar.copy()
    zero_moment_point[supported] -= (
        heights[supported] / (GRAVITY + vertical_accelerations[supported])
    )[:, None] * accelerations[supported]
    return CenterOfMassPlan(
        positions=np.concatenate((planar, heights[:, None]), axis=1),
        velocities=np.concatenate((velocities, vertical_velocities[:, None]), axis=1),
        accelerations=np.concatenate((accelerations, vertical_accelerations[:, None]), axis=1),
        zero_moment_point=zero_moment_point,
    )


__all__ = [
    "FLIGHT_LOAD",
    "GRAVITY",
    "CenterOfMassPlan",
    "PlannerWeights",
    "plan_center_of_mass",
    "plan_vertical",
]
