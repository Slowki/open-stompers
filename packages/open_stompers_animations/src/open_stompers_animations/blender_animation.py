"""Keyframe an Open Stompers rig inside Blender from exported animation arrays.

Run by Blender as `blender --background rig.blend --python blender_animation.py --
<animation.npz> <output.blend>`.
"""

import sys
from pathlib import Path

import bpy  # ty: ignore[unresolved-import]
import numpy as np


def _arguments() -> tuple[Path, Path]:
    """Return the animation arrays and output project paths."""
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    return Path(arguments[0]), Path(arguments[1])


def main() -> None:
    """Keyframe the root and joint bones, then save the project."""
    data_path, output_path = _arguments()
    data = np.load(data_path)
    name = str(data["name"])
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    scene = bpy.context.scene
    count = len(data["root_positions"])
    scene.render.fps = round(1.0 / float(data["sample_interval"]))
    scene.frame_start, scene.frame_end = 1, count
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    armature.animation_data_create()
    armature.animation_data.action = bpy.data.actions.new(name)
    armature.rotation_mode = "QUATERNION"
    bones = {
        str(joint): armature.pose.bones[str(joint)]
        for joint in data["joint_names"]
        if str(joint) in armature.pose.bones
    }
    columns = {str(joint): index for index, joint in enumerate(data["joint_names"])}
    for frame in range(count):
        number = frame + 1
        # The rig rest pose puts the root link at the armature origin.
        armature.location = data["root_positions"][frame]
        armature.rotation_quaternion = data["root_quaternions"][frame]
        armature.keyframe_insert("location", frame=number)
        armature.keyframe_insert("rotation_quaternion", frame=number)
        for joint, bone in bones.items():
            bone.rotation_mode = "XYZ"
            channel = int(bone.get("rotation_channel", 0))
            direction = float(bone.get("rotation_channel_direction", 1.0))
            offset = float(bone.get("coordinate_offset_radians", 0.0))
            angle = float(data["joint_positions"][frame, columns[joint]])
            bone.rotation_euler[channel] = direction * angle + offset
            bone.keyframe_insert("rotation_euler", index=channel, frame=number)
    armature["animation_name"] = name
    armature["animation_kind"] = str(data["kind"])
    armature["animation_commands"] = data["commands"].reshape(-1).tolist()
    scene.frame_set(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    main()
