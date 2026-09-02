"""Export animations as keyframed Blender projects."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from open_stompers_hardware_models import load
from open_stompers_hardware_models.export import BlenderExportError, ExportFormat, export_model

from .robots import parse_variant
from .types import Animation


def export_blender_animation(
    animation: Animation, output: str | Path, *, rig: str | Path | None = None
) -> Path:
    """Write a Blender project with the robot rig keyframed by an animation.

    Args:
        animation: Animation to keyframe.
        output: Destination `.blend` file.
        rig: Existing rigged `.blend` of the same robot; built from the hardware model when omitted.

    Returns:
        The written project path.

    Raises:
        BlenderExportError: If Blender is missing or fails.
    """
    if (blender := shutil.which("blender")) is None:
        raise BlenderExportError("Blender export requires the 'blender' executable")
    destination = Path(output)
    with tempfile.TemporaryDirectory(prefix="open-stompers-animation-") as temporary:
        directory = Path(temporary)
        rig_path = Path(rig) if rig is not None else directory / "rig.blend"
        if rig is None:
            export_model(load(parse_variant(animation.variant)), rig_path, ExportFormat.BLENDER)
        data_path = directory / "animation.npz"
        np.savez(
            data_path,
            name=np.array(animation.name),
            kind=np.array(str(animation.kind)),
            sample_interval=np.array(animation.sample_interval),
            joint_names=np.array(animation.joint_names),
            joint_positions=animation.joint_positions,
            root_positions=animation.root_positions,
            root_quaternions=animation.root_quaternions,
            commands=animation.commands,
        )
        script = Path(__file__).with_name("blender_animation.py")
        try:
            subprocess.run(
                [
                    blender,
                    "--background",
                    str(rig_path.resolve()),
                    "--python",
                    str(script),
                    "--",
                    str(data_path),
                    str(destination.resolve()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise BlenderExportError(f"Blender animation export failed:\n{error.stderr}") from error
    return destination


__all__ = ["export_blender_animation"]
