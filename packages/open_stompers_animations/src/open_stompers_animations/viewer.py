"""Rerun viewer for animations and their inverse dynamics."""

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.urdf import UrdfTree

from open_stompers_hardware_models import HardwareVariant, MicroduckVariant, load
from open_stompers_hardware_models.assets import cache_root as hardware_model_cache_root
from open_stompers_hardware_models.export import export_model
from open_stompers_hardware_models.robots import microduck, mini_pi_plus

from .analysis.inverse_dynamics import InverseDynamics
from .analysis.metrics import analysis_paths, read_dynamics
from .format import read_animation
from .robots import parse_variant
from .sampling import sample_animation
from .types import Animation, AnimationKind


def resolve_urdf_path(variant: HardwareVariant) -> Path:
    """Export the URDF of any hardware variant into the hardware model cache.

    Args:
        variant: Hardware variant to display.

    Returns:
        Path of the URDF with resolvable meshes.
    """
    match variant:
        case MicroduckVariant():
            microduck.ensure_meshes(variant)
        case _:
            mini_pi_plus.ensure_meshes(variant)
    destination = hardware_model_cache_root() / "rerun_models" / f"{variant}.urdf"
    # Always re-export so fixes to the hardware model never meet a stale cache.
    return export_model(load(variant), destination)


def _log_animation(
    animation: Animation,
    urdf: UrdfTree,
    dynamics: InverseDynamics | None,
    cycles: int,
) -> None:
    """Send one animation as columns on its own timeline."""
    repeats = cycles if animation.kind is AnimationKind.LOOP else 1
    count = animation.frame_count * repeats
    times = np.arange(count) * animation.sample_interval
    sample = sample_animation(animation, times)
    indexes = [rr.TimeColumn(animation.name, duration=times)]
    with warnings.catch_warnings():
        # Angles exactly at a limit can sit a float32 rounding error outside it.
        warnings.filterwarnings("ignore", message=".* is outside limits .*", category=UserWarning)
        for column, name in enumerate(animation.joint_names):
            if (joint := urdf.get_joint_by_name(name)) is None:
                continue
            rr.send_columns(
                f"robot/joints/{name}",
                indexes=indexes,
                columns=joint.compute_transform_columns(
                    sample.joint_positions[:, column].astype(float).tolist()
                ),
            )
    rr.send_columns(
        "world/base_transform",
        indexes=indexes,
        columns=rr.Transform3D.columns(
            translation=sample.root_positions,
            quaternion=sample.root_quaternions[:, [1, 2, 3, 0]],
            parent_frame=["world"] * count,
            child_frame=[urdf.root_link().name] * count,
        ),
    )
    rr.send_columns(
        "world/planned_center_of_mass",
        indexes=indexes,
        columns=rr.Points3D.columns(
            positions=sample.center_of_mass,
            radii=[0.012] * count,
            colors=[[255, 170, 0]] * count,
        ),
    )
    ground = np.concatenate((sample.zero_moment_point, np.zeros((count, 1))), axis=1)
    rr.send_columns(
        "world/planned_zero_moment_point",
        indexes=indexes,
        columns=rr.Points3D.columns(
            positions=ground, radii=[0.008] * count, colors=[[0, 160, 255]] * count
        ),
    )
    for axis, label in enumerate(("forward_velocity", "lateral_velocity", "yaw_rate")):
        rr.send_columns(
            f"commands/{label}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=sample.commands[:, axis]),
        )
    rr.send_columns(
        "gait/phase", indexes=indexes, columns=rr.Scalars.columns(scalars=sample.phases)
    )
    for column, name in enumerate(animation.contact_names):
        rr.send_columns(
            f"contacts/{name}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=sample.contacts[:, column].astype(np.float64)),
        )
    rr.send_columns(
        "gait/behavior",
        indexes=indexes,
        columns=rr.TextLog.columns(text=[str(behavior) for behavior in sample.behaviors]),
    )
    if dynamics is not None:
        _log_dynamics(animation, dynamics, repeats)


def _log_dynamics(animation: Animation, dynamics: InverseDynamics, repeats: int) -> None:
    """Send per-frame torques, contact forces, and the realized center of mass."""
    count = len(dynamics.times) * repeats
    indexes = [rr.TimeColumn(animation.name, duration=np.arange(count) * animation.sample_interval)]
    for column, name in enumerate(dynamics.joint_names):
        rr.send_columns(
            f"torques/{name}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=np.tile(dynamics.joint_torques[:, column], repeats)),
        )
    for column, name in enumerate(dynamics.contact_names):
        rr.send_columns(
            f"contact_forces/{name}",
            indexes=indexes,
            columns=rr.Scalars.columns(
                scalars=np.tile(dynamics.contact_forces[:, column, 2], repeats)
            ),
        )
    rr.send_columns(
        "residual/force",
        indexes=indexes,
        columns=rr.Scalars.columns(
            scalars=np.tile(np.linalg.norm(dynamics.base_residual[:, :3], axis=1), repeats)
        ),
    )
    if repeats == 1:
        rr.send_columns(
            "world/center_of_mass",
            indexes=indexes,
            columns=rr.Points3D.columns(
                positions=dynamics.center_of_mass,
                radii=[0.012] * count,
                colors=[[255, 70, 30]] * count,
            ),
        )


def view_animations(
    paths: Sequence[Path],
    *,
    cycles: int = 3,
    save: Path | None = None,
) -> None:
    """Open animation files in Rerun, one timeline per animation.

    Inverse dynamics written by the analysis next to an animation are shown too.

    Args:
        paths: Animation Parquet files of the same hardware variant.
        cycles: Loop repetitions to play.
        save: Recording file to write instead of spawning a viewer.
    """
    animations = [read_animation(path) for path in paths]
    if not animations:
        raise ValueError("Provide at least one animation")
    if len({animation.variant for animation in animations}) != 1:
        raise ValueError("All animations must belong to the same hardware variant")
    rr.init("open_stompers_animations", spawn=save is None)
    if save is not None:
        rr.save(save)
    rr.set_log_time_enabled(False)
    # Hang the named world frame off the root entity frame that 3D views target.
    rr.log("world", rr.Transform3D(parent_frame="tf#/", child_frame="world"), static=True)
    for entity in ("world/planned_center_of_mass", "world/planned_zero_moment_point"):
        rr.log(entity, rr.CoordinateFrame("world"), static=True)
    rr.log("world/center_of_mass", rr.CoordinateFrame("world"), static=True)
    urdf = UrdfTree.from_file_path(
        resolve_urdf_path(parse_variant(animations[0].variant)),
        entity_path_prefix="robot",
        static_transform_entity_path="robot/tf_static",
    )
    urdf.log_urdf_to_recording()
    for path, animation in zip(paths, animations, strict=True):
        _, frames_path = analysis_paths(path)
        dynamics = read_dynamics(frames_path) if frames_path.exists() else None
        _log_animation(animation, urdf, dynamics, cycles)


__all__ = ["resolve_urdf_path", "view_animations"]
