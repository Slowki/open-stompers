"""Build a Blender armature and rigid link attachments from an export manifest."""

import json
import sys
from pathlib import Path
from typing import Any

import bpy  # ty: ignore[unresolved-import]
from mathutils import Euler, Matrix, Vector  # ty: ignore[unresolved-import]


def _arguments() -> tuple[Path, Path, Path]:
    """Read the scene, manifest, and output paths."""
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 3:
        raise ValueError("Expected a GLB path, manifest path, and Blender output path")
    return Path(arguments[0]), Path(arguments[1]), Path(arguments[2])


def _transform(xyz: list[float], rpy: list[float]) -> Matrix:
    """Build a homogeneous transform from XYZ/RPY values."""
    result = Euler(rpy, "XYZ").to_matrix().to_4x4()
    result.translation = Vector(xyz)
    return result


def _motion(joint: dict[str, Any], angle: float) -> Matrix:
    """Build one joint's rest offset transform."""
    if joint["type"] != "revolute" or joint["axis"] is None:
        return Matrix.Identity(4)
    axis = Vector(joint["axis"]).normalized()
    return Matrix.Rotation(angle, 4, axis)


def _link_frames(manifest: dict[str, Any], angles: dict[str, float]) -> dict[str, Matrix]:
    """Compute link frames for a joint configuration."""
    frames = {manifest["root_link"]: Matrix.Identity(4)}
    joints_by_child = {joint["child"]: joint for joint in manifest["joints"]}
    for link in manifest["links"]:
        if link["parent"] is None:
            continue
        joint = joints_by_child[link["name"]]
        frames[link["name"]] = (
            frames[joint["parent"]]
            @ _transform(joint["xyz"], joint["rpy"])
            @ _motion(joint, angles.get(joint["name"], 0.0))
        )
    return frames


def _bone_frame(frame: Matrix, joint: dict[str, Any]) -> Matrix:
    """Align the bone's local X axis to its revolute joint axis."""
    if joint["type"] != "revolute" or joint["axis"] is None:
        return frame.copy()
    orientation = Vector(joint["axis"]).normalized().to_track_quat("X", "Z")
    result = frame @ orientation.to_matrix().to_4x4()
    result.translation = frame.translation
    return result


def _joint_bone_direction(frame: Matrix, axis: Vector) -> tuple[Vector, Vector]:
    """Choose a bone direction and roll that put the joint axis on local X."""
    bone_direction = frame.to_3x3() @ Vector((0.0, 1.0, 0.0))
    bone_direction -= axis * bone_direction.dot(axis)
    if bone_direction.length_squared <= 1e-10:
        reference = Vector((0.0, 0.0, 1.0))
        if abs(axis.dot(reference)) > 0.9:
            reference = Vector((0.0, 1.0, 0.0))
        bone_direction = axis.cross(reference)
    bone_direction.normalize()
    bone_roll_axis = axis.cross(bone_direction).normalized()
    return bone_direction, bone_roll_axis


def _make_armature(  # noqa: PLR0912, PLR0915
    manifest: dict[str, Any],
) -> tuple[bpy.types.Object, dict[str, str]]:
    """Create a joint hierarchy and limb targets."""
    zero_frames = _link_frames(manifest, {})
    frames = zero_frames
    range_midpoints = {
        joint["name"]: (joint["lower"] + joint["upper"]) / 2
        if joint["lower"] is not None and joint["upper"] is not None
        else 0.0
        for joint in manifest["joints"]
    }
    joints_by_child = {joint["child"]: joint for joint in manifest["joints"]}
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = f"{manifest['name']}_Rig"
    armature.data.name = f"{manifest['name']}_Armature"
    edit_bones = armature.data.edit_bones
    edit_bones.remove(edit_bones[0])

    root_control = edit_bones.new("Master_Control")
    root_control.head = (0.0, 0.0, 0.0)
    root_control.tail = (0.0, 0.0, 0.15)
    body_control = edit_bones.new("Body_Root_Control")
    body_control.head = (0.0, 0.0, 0.0)
    body_control.tail = (0.0, 0.0, 0.12)
    body_control.parent = root_control

    bone_for_link = {manifest["root_link"]: "Body_Root"}
    root_link_bone = edit_bones.new("Body_Root")
    root_link_bone.head = frames[manifest["root_link"]].translation
    root_link_bone.tail = root_link_bone.head + Vector((0.0, 0.0, 0.1))
    root_link_bone.parent = body_control

    for link in manifest["links"]:
        if link["parent"] is None:
            continue
        joint = joints_by_child[link["name"]]
        bone_name = joint["name"]
        bone = edit_bones.new(bone_name)
        bone.parent = edit_bones[bone_for_link[joint["parent"]]]
        bone.use_connect = False
        if joint["type"] == "revolute" and joint["axis"] is not None:
            rest_axis = (
                frames[joint["parent"]] @ _transform(joint["xyz"], joint["rpy"])
            ).to_3x3() @ Vector(joint["axis"]).normalized()
            bone_direction, bone_roll_axis = _joint_bone_direction(
                frames[link["name"]], rest_axis.normalized()
            )
            bone.head = frames[link["name"]].translation
            bone.tail = bone.head + bone_direction * 0.035
            bone.align_roll(bone_roll_axis)
        else:
            bone.matrix = frames[link["name"]]
            bone.length = 0.035
        bone_for_link[link["name"]] = bone_name

    armature.data.pose_position = "POSE"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    armature.data.display_type = "OCTAHEDRAL"
    armature["robot_model"] = manifest["name"]
    armature["joint_units"] = "radians"

    for joint in manifest["joints"]:
        if joint["type"] == "fixed":
            bone = armature.pose.bones[joint["name"]]
            bone.lock_location = (True, True, True)
            bone.lock_rotation = (True, True, True)
            bone.lock_scale = (True, True, True)
            bone.lock_ik_x = True
            bone.lock_ik_y = True
            bone.lock_ik_z = True
            bone.rotation_mode = "XYZ"
            continue
        if joint["type"] != "revolute":
            continue
        bone = armature.pose.bones[joint["name"]]
        bone.rotation_mode = "XYZ"
        rotation_channel = 0
        channel_direction = 1.0
        bone.rotation_euler[rotation_channel] = 0.0
        bone.lock_location = (True, True, True)
        bone.lock_rotation = tuple(index != rotation_channel for index in range(3))
        bone.lock_scale = (True, True, True)
        bone.lock_ik_x = False
        bone.lock_ik_y = True
        bone.lock_ik_z = True
        bone["joint_type"] = joint["type"]
        bone["coordinate_offset_radians"] = 0.0
        bone["rotation_channel"] = rotation_channel
        bone["rotation_channel_direction"] = channel_direction
        if joint["lower"] is not None and joint["upper"] is not None:
            bone["joint_lower_radians"] = joint["lower"]
            bone["joint_upper_radians"] = joint["upper"]
            bone["joint_range_midpoint_radians"] = range_midpoints[joint["name"]]
            ik_limit_names = ("use_ik_limit_x", "use_ik_limit_y", "use_ik_limit_z")
            ik_minimum_names = ("ik_min_x", "ik_min_y", "ik_min_z")
            ik_maximum_names = ("ik_max_x", "ik_max_y", "ik_max_z")
            setattr(bone, ik_limit_names[rotation_channel], True)
            setattr(bone, ik_minimum_names[rotation_channel], joint["lower"])
            setattr(bone, ik_maximum_names[rotation_channel], joint["upper"])
            limit = bone.constraints.new("LIMIT_ROTATION")
            limit.name = f"{joint['name']} Mechanical Limits"
            limit.owner_space = "LOCAL"
            limit_names = ("use_limit_x", "use_limit_y", "use_limit_z")
            minimum_names = ("min_x", "min_y", "min_z")
            maximum_names = ("max_x", "max_y", "max_z")
            for index in range(3):
                setattr(limit, limit_names[index], True)
                if index == rotation_channel:
                    setattr(limit, minimum_names[index], joint["lower"])
                    setattr(limit, maximum_names[index], joint["upper"])
                else:
                    setattr(limit, minimum_names[index], 0.0)
                    setattr(limit, maximum_names[index], 0.0)

    all_bones = armature.data.collections.new("Joint Bones")
    controls = armature.data.collections.new("Animation Controls")
    helpers = armature.data.collections.new("Fixed Links")
    fixed_joint_names = {joint["name"] for joint in manifest["joints"] if joint["type"] == "fixed"}
    target_names: dict[str, str] = {}
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    for limb in manifest["limbs"]:
        endpoint_joint = joints_by_child[limb["endpoint_link"]]
        endpoint_frame = _bone_frame(zero_frames[limb["endpoint_link"]], endpoint_joint)
        target_name = f"{limb['name']}_Target"
        target = edit_bones.new(target_name)
        target.matrix = endpoint_frame
        target.length = 0.11
        target.parent = edit_bones["Master_Control"]
        target_names[limb["name"]] = target_name

        base_joint = next(
            joint for joint in manifest["joints"] if joint["name"] == limb["root_joint"]
        )
        base_frame = zero_frames[base_joint["parent"]] @ _transform(
            base_joint["xyz"], base_joint["rpy"]
        )
        line = zero_frames[limb["endpoint_link"]].translation - base_frame.translation
        interior_links = [
            next(joint for joint in manifest["joints"] if joint["name"] == joint_name)["child"]
            for joint_name in limb["joints"][:-1]
        ]
        midpoint_position = zero_frames[interior_links[len(interior_links) // 2]].translation
        bend_direction = midpoint_position - base_frame.translation
        if line.length_squared > 1e-10:
            bend_direction -= line * (bend_direction.dot(line) / line.length_squared)
        if bend_direction.length_squared <= 1e-10:
            reference = Vector((1.0, 0.0, 0.0))
            if abs(line.normalized().dot(reference)) > 0.9:
                reference = Vector((0.0, 1.0, 0.0))
            bend_direction = line.cross(reference)
        bend_direction.normalize()
        pole_name = f"{limb['name']}_Pole"
        pole = edit_bones.new(pole_name)
        pole.head = midpoint_position + bend_direction * 0.15
        pole.tail = pole.head + Vector((0.0, 0.08, 0.0))
        pole.parent = edit_bones["Master_Control"]
        target_names[f"{limb['name']}_pole"] = pole_name
    bpy.ops.object.mode_set(mode="OBJECT")

    for bone in armature.data.bones:
        if bone.name.endswith(("_Control", "_Target", "_Pole")):
            controls.assign(bone)
        elif bone.name == "Body_Root" or bone.name in fixed_joint_names:
            helpers.assign(bone)
        else:
            all_bones.assign(bone)

    for limb in manifest["limbs"]:
        endpoint_bone = bone_for_link[limb["endpoint_link"]]
        constraint = armature.pose.bones[endpoint_bone].constraints.new("IK")
        constraint.name = f"{limb['name']} Target IK"
        constraint.target = armature
        constraint.subtarget = target_names[limb["name"]]
        constraint.pole_target = armature
        constraint.pole_subtarget = target_names[f"{limb['name']}_pole"]
        constraint.chain_count = len(limb["joints"])
        constraint.use_tail = False
        constraint.use_stretch = False
        constraint.use_rotation = limb["orientation_weight"] > 0
        constraint.weight = 1.0
        constraint.orient_weight = limb["orientation_weight"]
        # The source zero pose can lie outside a joint's physical limits. Keep
        # it intact on open; animators can enable each target after matching it.
        armature[f"ik_{limb['name']}"] = 0.0
        armature.id_properties_ui(f"ik_{limb['name']}").update(
            min=0.0,
            max=1.0,
            description="Blend from direct joint posing (0) to target posing (1).",
        )
        driver = constraint.driver_add("influence").driver
        driver.type = "SCRIPTED"
        driver.expression = "blend"
        variable = driver.variables.new()
        variable.name = "blend"
        variable.type = "SINGLE_PROP"
        variable.targets[0].id = armature
        variable.targets[0].data_path = f'["ik_{limb["name"]}"]'
    bpy.context.view_layer.update()
    for limb in manifest["limbs"]:
        endpoint_bone = bone_for_link[limb["endpoint_link"]]
        target = armature.pose.bones[target_names[limb["name"]]]
        target.matrix = armature.pose.bones[endpoint_bone].matrix.copy()
    bpy.context.view_layer.update()
    default_collection = armature.data.collections.get("Bones")
    if default_collection is not None and not default_collection.bones:
        armature.data.collections.remove(default_collection)
    return armature, bone_for_link


def _attach_visuals(
    armature: bpy.types.Object, manifest: dict[str, Any], bone_for_link: dict[str, str]
) -> None:
    """Parent each imported rigid mesh to its corresponding link bone."""
    visuals_by_node = {visual["node"]: visual for visual in manifest["visuals"]}
    link_frames = _link_frames(manifest, {})
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for mesh in meshes:
        visual = visuals_by_node.get(mesh.name)
        if visual is None:
            visual = visuals_by_node.get(mesh.data.name)
        if visual is None:
            continue
        world_transform = mesh.matrix_world.copy()
        bone_name = bone_for_link[visual["link"]]
        bone_transform = armature.matrix_world @ armature.data.bones[bone_name].matrix_local
        # GLTF export bakes each visual's world-space pose into its vertices.
        # Move the vertices into the bone's rest frame before making it a child.
        mesh.data = mesh.data.copy()
        mesh.data.transform(
            bone_transform.inverted() @ link_frames[visual["link"]] @ world_transform
        )
        mesh.name = visual["name"]
        mesh.parent = armature
        mesh.parent_type = "BONE"
        mesh.parent_bone = bone_name
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_world = bone_transform
        mesh["robot_link"] = visual["link"]


def _organize_scene(manifest: dict[str, Any], armature: bpy.types.Object) -> None:
    """Name the robot collection and select its armature for posing."""
    collection = bpy.data.collections.new(f"{manifest['name']} Robot")
    bpy.context.scene.collection.children.link(collection)
    for obj in list(bpy.context.scene.objects):
        if obj == armature:
            continue
        for owner in tuple(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
    for owner in tuple(armature.users_collection):
        owner.objects.unlink(armature)
    collection.objects.link(armature)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.length_unit = "METERS"
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.context.scene.frame_set(1)


def main() -> None:
    """Import the robot scene, build its rig, and save a Blender project."""
    glb_path, manifest_path, output_path = _arguments()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    armature, bone_for_link = _make_armature(manifest)
    _attach_visuals(armature, manifest, bone_for_link)
    _organize_scene(manifest, armature)
    text = bpy.data.texts.new("Open Stompers Rig Guide")
    text.write(
        "Open the Animation Controls bone collection and move the hand or foot targets. "
        "Set the armature's ik_<limb name> property to 1 to enable each limb target, or leave "
        "it at 0 to pose the named joint bones directly. Target matching is manual when "
        "switching modes. Keyframe target bones, joint rotations, or the "
        "ik_<limb name> properties. "
        "The Master_Control moves the robot and targets together; Body_Root_Control moves "
        "the robot relative to those targets. Joint coordinates use radians.\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    main()
