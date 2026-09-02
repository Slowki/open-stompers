"""Summary metrics of animations and their inverse dynamics."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ..model import build_robot_model
from ..planning.center_of_mass import GRAVITY
from ..robots import parse_variant
from ..types import Animation, AnimationKind
from .inverse_dynamics import InverseDynamics, inverse_dynamics

#: Speed below which the cost of transport is not reported, in meters per second.
MINIMUM_TRANSPORT_SPEED: Final = 0.05


@dataclass(frozen=True)
class JointMetrics:
    """Load and range usage of one joint."""

    #: Largest absolute torque in newton meters.
    peak_torque: float
    #: Root mean square torque in newton meters.
    root_mean_square_torque: float
    #: Peak torque divided by the effort limit.
    peak_torque_ratio: float
    #: Largest absolute speed in radians per second.
    peak_velocity: float
    #: Peak speed divided by the speed limit.
    peak_velocity_ratio: float
    #: Smallest distance to a position limit in radians.
    minimum_limit_margin: float


@dataclass(frozen=True)
class CenterOfMassMetrics:
    """Center of mass motion statistics."""

    #: Mean height above the ground in meters.
    height_mean: float
    #: Lowest height in meters.
    height_minimum: float
    #: Highest height in meters.
    height_maximum: float
    #: Height standard deviation in meters.
    height_standard_deviation: float
    #: Peak-to-peak lateral sway relative to the feet midpoint, in meters.
    lateral_sway: float
    #: Largest speed in meters per second.
    peak_speed: float
    #: Largest acceleration in meters per second squared.
    peak_acceleration: float
    #: Largest distance between the planned and the realized center of mass, in meters.
    planned_deviation: float


@dataclass(frozen=True)
class AnimationMetrics:
    """Physical feasibility and effort summary of an animation."""

    #: Animation name.
    name: str
    #: Hardware variant.
    variant: str
    #: Loop or action.
    kind: str
    #: Playback duration in seconds.
    duration: float
    #: Total mass in kilograms.
    mass: float
    #: Mean commanded `[vx, vy, yaw_rate]`.
    mean_command: list[float]
    #: Realized mean heading-frame `[vx, vy, yaw_rate]`.
    mean_velocity: list[float]
    #: Per-joint metrics keyed by joint name.
    joints: dict[str, JointMetrics]
    #: Joint with the highest peak torque ratio.
    limiting_joint: str
    #: Highest peak torque ratio over all joints.
    peak_torque_ratio: float
    #: Highest peak speed ratio over all joints.
    peak_velocity_ratio: float
    #: Largest total mechanical power in watts.
    peak_mechanical_power: float
    #: Mean positive mechanical power in watts.
    mean_positive_power: float
    #: Positive mechanical work per weight and distance, if the robot travels.
    cost_of_transport: float | None
    #: Largest root force the contacts cannot supply, excluding contact changes, in newtons.
    steady_residual_force: float
    #: Largest root torque the contacts cannot supply, excluding contact changes, in N m.
    steady_residual_torque: float
    #: Largest unsupplied root force within a frame of a contact change, in newtons.
    transition_residual_force: float
    #: Steady residual force as a fraction of the robot weight.
    steady_residual_ratio: float
    #: Largest tangential to normal force ratio of a loaded foot.
    required_friction: float
    #: Smallest signed distance of the center of pressure inside the support polygon.
    minimum_support_margin: float
    #: Largest speed of a planted sole in meters per second.
    maximum_foot_slip: float
    #: Deepest self-collision penetration in meters.
    maximum_self_collision_depth: float
    #: Mean knee bend of loaded legs in radians, zero for straight legs.
    stance_knee_flexion_mean: float
    #: Least knee bend of loaded legs in radians.
    stance_knee_flexion_minimum: float
    #: Center of mass statistics.
    center_of_mass: CenterOfMassMetrics

    def to_dict(self) -> dict[str, object]:
        """Convert the metrics to JSON-compatible values."""
        return asdict(self)


def _headings(animation: Animation) -> NDArray[np.float64]:
    """Return the base heading of every frame in radians."""
    quaternions = animation.root_quaternions.astype(np.float64)
    return np.arctan2(
        2 * (quaternions[:, 0] * quaternions[:, 3] + quaternions[:, 1] * quaternions[:, 2]),
        1 - 2 * (quaternions[:, 2] ** 2 + quaternions[:, 3] ** 2),
    )


def _heading_velocity(animation: Animation) -> NDArray[np.float64]:
    """Return the mean heading-frame velocity of the planned center of mass."""
    positions = animation.center_of_mass.astype(np.float64)
    headings = np.unwrap(_headings(animation))
    interval = animation.sample_interval
    if animation.kind is AnimationKind.LOOP:
        offset = animation.loop_offset
        following = offset.apply_positions(positions[:1])
        preceding = offset.inverse().apply_positions(positions[-1:])
        padded = np.concatenate((preceding, positions, following))
        planar = (padded[2:, :2] - padded[:-2, :2]) / (2 * interval)
        rate = offset.yaw / animation.duration
    else:
        planar = np.gradient(positions[:, :2], interval, axis=0)
        rate = float(np.gradient(headings, interval).mean())
    cosine, sine = np.cos(headings), np.sin(headings)
    forward = cosine * planar[:, 0] + sine * planar[:, 1]
    lateral = -sine * planar[:, 0] + cosine * planar[:, 1]
    return np.array([forward.mean(), lateral.mean(), rate])


def _transitions(contacts: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Flag frames within one frame of a planned contact change."""
    changed = np.zeros(len(contacts), dtype=bool)
    switches = np.any(contacts[1:] != contacts[:-1], axis=1)
    changed[1:] |= switches
    changed[:-1] |= switches
    return changed


def _lateral_sway(animation: Animation, dynamics: InverseDynamics) -> float:
    """Measure the peak-to-peak sideways offset of the center of mass from the feet midpoint."""
    headings = _headings(animation)
    midpoint = dynamics.sole_positions[:, :, :2].mean(axis=1)
    offset = dynamics.center_of_mass[:, :2] - midpoint
    lateral = -np.sin(headings) * offset[:, 0] + np.cos(headings) * offset[:, 1]
    return float(np.ptp(lateral))


def summarize(animation: Animation, dynamics: InverseDynamics) -> AnimationMetrics:
    """Summarize the effort and feasibility of an animation.

    Args:
        animation: Analyzed animation.
        dynamics: Its inverse dynamics.

    Returns:
        The summary metrics.
    """
    robot_model = build_robot_model(parse_variant(animation.variant))
    columns = [robot_model.joint_index(name) for name in animation.joint_names]
    efforts = robot_model.effort_limits[columns]
    speeds = robot_model.velocity_limits[columns]
    lower, upper = robot_model.lower_limits[columns], robot_model.upper_limits[columns]
    torques, velocities = np.abs(dynamics.joint_torques), np.abs(dynamics.joint_velocities)
    positions = animation.joint_positions.astype(np.float64)
    margins = np.minimum(positions - lower, upper - positions).min(axis=0)
    joints = {
        name: JointMetrics(
            peak_torque=float(torques[:, index].max()),
            root_mean_square_torque=float(np.sqrt(np.mean(dynamics.joint_torques[:, index] ** 2))),
            peak_torque_ratio=float(torques[:, index].max() / efforts[index]),
            peak_velocity=float(velocities[:, index].max()),
            peak_velocity_ratio=float(velocities[:, index].max() / speeds[index]),
            minimum_limit_margin=float(margins[index]),
        )
        for index, name in enumerate(animation.joint_names)
    }
    limiting = max(joints, key=lambda name: joints[name].peak_torque_ratio)
    weight = robot_model.mass * GRAVITY
    power = dynamics.mechanical_power
    positive = float(np.clip(power, 0, None).mean())
    velocity = _heading_velocity(animation)
    speed = float(np.linalg.norm(velocity[:2]))
    transition = _transitions(animation.contacts)
    if animation.kind is AnimationKind.ACTION:
        # End frames of cropped actions only have extrapolated neighbours.
        transition[[0, -1]] = True
    steady = ~transition
    forces = np.linalg.norm(dynamics.base_residual[:, :3], axis=1)
    moments = np.linalg.norm(dynamics.base_residual[:, 3:], axis=1)
    feet = dynamics.contact_forces[:, :2]
    loaded = feet[..., 2] > 0.05 * weight
    friction = np.linalg.norm(feet[..., :2], axis=-1) / np.maximum(feet[..., 2], 1e-9)
    center_of_mass = dynamics.center_of_mass
    interval = animation.sample_interval
    center_velocity = np.gradient(center_of_mass, interval, axis=0)
    center_acceleration = np.gradient(center_velocity, interval, axis=0)
    trimmed = slice(1, -1) if animation.kind is AnimationKind.ACTION else slice(None)
    knees = np.concatenate(
        [
            sign * positions[animation.contacts[:, side], animation.joint_names.index(name)]
            for side, (name, sign) in enumerate(robot_model.robot.knee_flexion.items())
        ]
    )
    return AnimationMetrics(
        name=animation.name,
        variant=animation.variant,
        kind=str(animation.kind),
        duration=animation.duration,
        mass=robot_model.mass,
        mean_command=animation.commands.mean(axis=0).astype(float).tolist(),
        mean_velocity=velocity.astype(float).tolist(),
        joints=joints,
        limiting_joint=limiting,
        peak_torque_ratio=joints[limiting].peak_torque_ratio,
        peak_velocity_ratio=max(metrics.peak_velocity_ratio for metrics in joints.values()),
        peak_mechanical_power=float(np.abs(power).max()),
        mean_positive_power=positive,
        cost_of_transport=positive / (weight * speed) if speed > MINIMUM_TRANSPORT_SPEED else None,
        steady_residual_force=float(forces[steady].max(initial=0.0)),
        steady_residual_torque=float(moments[steady].max(initial=0.0)),
        transition_residual_force=float(forces[transition].max(initial=0.0)),
        steady_residual_ratio=float(forces[steady].max(initial=0.0) / weight),
        required_friction=float(friction[loaded].max(initial=0.0)),
        minimum_support_margin=float(np.nanmin(dynamics.support_margin, initial=np.inf)),
        maximum_foot_slip=float(dynamics.foot_slip.max()),
        maximum_self_collision_depth=float(dynamics.self_collision_depth.max()),
        stance_knee_flexion_mean=float(knees.mean()) if len(knees) else 0.0,
        stance_knee_flexion_minimum=float(knees.min()) if len(knees) else 0.0,
        center_of_mass=CenterOfMassMetrics(
            height_mean=float(center_of_mass[:, 2].mean()),
            height_minimum=float(center_of_mass[:, 2].min()),
            height_maximum=float(center_of_mass[:, 2].max()),
            height_standard_deviation=float(center_of_mass[:, 2].std()),
            lateral_sway=_lateral_sway(animation, dynamics),
            peak_speed=float(np.linalg.norm(center_velocity[trimmed], axis=1).max()),
            peak_acceleration=float(np.linalg.norm(center_acceleration[trimmed], axis=1).max()),
            planned_deviation=float(
                np.linalg.norm(center_of_mass - animation.center_of_mass, axis=1).max()
            ),
        ),
    )


def dynamics_table(dynamics: InverseDynamics) -> pa.Table:
    """Build a per-frame Parquet table of inverse dynamics results.

    Args:
        dynamics: Inverse dynamics to encode.

    Returns:
        One row per frame with joint and contact columns.
    """

    def vectors(values: NDArray) -> pa.FixedSizeListArray:
        flat = pa.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1))
        return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])

    columns = {
        "time": pa.array(dynamics.times),
        "joint_torque": vectors(dynamics.joint_torques),
        "joint_velocity": vectors(dynamics.joint_velocities),
        "contact_force": vectors(dynamics.contact_forces.reshape(len(dynamics.times), -1)),
        "center_of_pressure": vectors(dynamics.center_of_pressure),
        "support_margin": pa.array(dynamics.support_margin),
        "base_residual": vectors(dynamics.base_residual),
        "center_of_mass": vectors(dynamics.center_of_mass),
        "sole_positions": vectors(dynamics.sole_positions.reshape(len(dynamics.times), -1)),
        "foot_slip": vectors(dynamics.foot_slip),
        "mechanical_power": pa.array(dynamics.mechanical_power),
        "self_collision_depth": pa.array(dynamics.self_collision_depth),
    }
    header = {
        "joint_names": list(dynamics.joint_names),
        "contact_names": list(dynamics.contact_names),
    }
    return pa.table(columns).replace_schema_metadata(
        {b"open_stompers_analysis": json.dumps(header).encode()}
    )


def analysis_paths(animation_path: str | Path) -> tuple[Path, Path]:
    """Return the metrics JSON and per-frame Parquet paths beside an animation file."""
    path = Path(animation_path)
    stem = path.name.removesuffix(".parquet")
    return path.with_name(f"{stem}.analysis.json"), path.with_name(f"{stem}.analysis.parquet")


def analyze_animation(
    animation: Animation, animation_path: str | Path | None = None
) -> tuple[AnimationMetrics, InverseDynamics]:
    """Run inverse dynamics, summarize it, and optionally store both beside the animation.

    Args:
        animation: Animation to analyze.
        animation_path: Animation file whose sibling analysis files are written, if any.

    Returns:
        The summary metrics and the per-frame inverse dynamics.
    """
    dynamics = inverse_dynamics(animation)
    metrics = summarize(animation, dynamics)
    if animation_path is not None:
        summary_path, frames_path = analysis_paths(animation_path)
        summary_path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
        pq.write_table(dynamics_table(dynamics), frames_path, compression="zstd")
    return metrics, dynamics


def read_dynamics(path: str | Path) -> InverseDynamics:
    """Read per-frame inverse dynamics written by `analyze_animation`.

    Args:
        path: Analysis Parquet file.

    Returns:
        The decoded inverse dynamics.
    """
    table = pq.read_table(path)
    header = json.loads((table.schema.metadata or {})[b"open_stompers_analysis"])

    def values(name: str) -> NDArray[np.float64]:
        column = table.column(name).combine_chunks()
        if pa.types.is_fixed_size_list(column.type):
            width = column.type.list_size
            return column.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)
        return column.to_numpy(zero_copy_only=False)

    contacts = len(header["contact_names"])
    return InverseDynamics(
        times=values("time"),
        joint_names=tuple(header["joint_names"]),
        contact_names=tuple(header["contact_names"]),
        joint_velocities=values("joint_velocity"),
        joint_torques=values("joint_torque"),
        contact_forces=values("contact_force").reshape(-1, contacts, 3),
        center_of_pressure=values("center_of_pressure"),
        support_margin=values("support_margin"),
        base_residual=values("base_residual"),
        center_of_mass=values("center_of_mass"),
        sole_positions=values("sole_positions").reshape(-1, 2, 3),
        foot_slip=values("foot_slip"),
        mechanical_power=values("mechanical_power"),
        self_collision_depth=values("self_collision_depth"),
    )


__all__ = [
    "AnimationMetrics",
    "CenterOfMassMetrics",
    "JointMetrics",
    "analysis_paths",
    "analyze_animation",
    "dynamics_table",
    "read_dynamics",
    "summarize",
]
