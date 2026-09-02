//! Shared nalgebra and Rerun helpers for generated link transforms.
//!
//! Platform crates generate the kinematic constants. This module owns the
//! transform algebra and the Rerun logging helpers those modules call.

use std::path::{Path, PathBuf};

use nalgebra::{Isometry3, Quaternion, Unit, UnitQuaternion, Vector3};
use thiserror::Error;
pub use uom::si::f64::Angle;

/// Minimum and maximum positions for a revolute joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimits {
    /// Minimum position the joint can reach.
    pub minimum_position: Angle,
    /// Maximum position the joint can reach.
    pub maximum_position: Angle,
}

/// Create an `f64` UOM angle from a value in radians.
#[macro_export]
macro_rules! angle {
    ($radians:expr) => {
        $crate::transforms::Angle {
            dimension: ::std::marker::PhantomData,
            units: ::std::marker::PhantomData,
            value: $radians,
        }
    };
}

/// Parent-relative Rerun transform for one link or visual entity.
#[derive(Clone, Debug)]
pub struct RerunLinkTransform {
    /// Entity path relative to the caller's chosen robot prefix.
    pub entity_path: &'static str,
    /// Transform from the parent entity to this entity.
    pub transform: rerun::Transform3D,
}

/// Static mesh visual logged as a child of a link entity.
#[derive(Clone, Copy, Debug)]
pub struct VisualAsset {
    /// Entity path of the mesh, relative to the robot prefix.
    pub entity_path: &'static str,
    /// Cache-root-relative mesh path from the hardware model.
    pub mesh_resource: &'static str,
}

/// Failure while logging a generated robot model to Rerun.
#[derive(Debug, Error)]
pub enum TransformLogError {
    /// A visual mesh could not be read from the hardware-model cache.
    #[error("failed to read mesh {path}: {source}")]
    ReadMesh {
        /// Filesystem path that was opened.
        path: PathBuf,
        /// Filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// Rerun rejected a logged component.
    #[error(transparent)]
    Recording(#[from] rerun::RecordingStreamError),
}

/// Rigid transform from a translation and an `xyzw` quaternion.
#[must_use]
pub fn origin(translation: [f64; 3], rotation_xyzw: [f64; 4]) -> Isometry3<f64> {
    Isometry3::from_parts(
        nalgebra::Translation3::new(translation[0], translation[1], translation[2]),
        UnitQuaternion::new_normalize(Quaternion::new(
            rotation_xyzw[3],
            rotation_xyzw[0],
            rotation_xyzw[1],
            rotation_xyzw[2],
        )),
    )
}

/// URDF revolute joint transform: origin, then rotation about `axis` by `angle` radians.
#[must_use]
pub fn revolute(
    translation: [f64; 3],
    rotation_xyzw: [f64; 4],
    axis: [f64; 3],
    angle: f64,
) -> Isometry3<f64> {
    origin(translation, rotation_xyzw) * rotation_about(axis, angle)
}

/// URDF prismatic joint transform: origin, then translation along `axis` by `distance` meters.
#[must_use]
pub fn prismatic(
    translation: [f64; 3],
    rotation_xyzw: [f64; 4],
    axis: [f64; 3],
    distance: f64,
) -> Isometry3<f64> {
    origin(translation, rotation_xyzw)
        * Isometry3::from_parts(
            nalgebra::Translation3::new(axis[0] * distance, axis[1] * distance, axis[2] * distance),
            UnitQuaternion::identity(),
        )
}

/// URDF fixed joint transform.
#[must_use]
pub fn fixed(translation: [f64; 3], rotation_xyzw: [f64; 4]) -> Isometry3<f64> {
    origin(translation, rotation_xyzw)
}

/// Compose parent-relative link transforms into root-frame poses.
///
/// `parents[index]` is the parent link index. The root entry is unused.
/// Parents must precede children.
#[must_use]
pub fn world_from_local<const N: usize>(
    local: &[Isometry3<f64>; N],
    parents: &[usize; N],
) -> [Isometry3<f64>; N] {
    let mut world = [Isometry3::identity(); N];
    for index in 1..N {
        debug_assert!(parents[index] < index, "link parents must precede children");
        world[index] = world[parents[index]] * local[index];
    }
    world
}

/// Convert a rigid transform and mesh scale into a Rerun `Transform3D`.
#[must_use]
pub fn to_rerun(pose: Isometry3<f64>, scale: [f64; 3]) -> rerun::Transform3D {
    let translation = pose.translation.vector;
    let rotation = pose.rotation;
    let mut transform = rerun::Transform3D::from_translation_rotation(
        [
            translation.x as f32,
            translation.y as f32,
            translation.z as f32,
        ],
        rerun::Quaternion::from_xyzw([
            rotation.i as f32,
            rotation.j as f32,
            rotation.k as f32,
            rotation.w as f32,
        ]),
    );
    if scale != [1.0, 1.0, 1.0] {
        transform = transform.with_scale(rerun::Scale3D(rerun::Vec3D([
            scale[0] as f32,
            scale[1] as f32,
            scale[2] as f32,
        ])));
    }
    transform
}

/// Join a caller prefix with a generated entity path.
#[must_use]
pub fn prefixed_entity_path(prefix: &str, entity_path: &str) -> String {
    if prefix.is_empty() {
        entity_path.to_owned()
    } else {
        format!("{prefix}/{entity_path}")
    }
}

/// Root directory of the hardware-model mesh cache.
///
/// Matches `open_stompers_hardware_models`: `HARDWARE_MODELS_CACHE_ROOT` overrides
/// the cache, otherwise meshes live under `$XDG_CACHE_HOME/openstompers` or
/// `~/.cache/openstompers`.
#[must_use]
pub fn hardware_model_cache_root() -> PathBuf {
    hardware_model_cache_root_from(
        std::env::var_os("HARDWARE_MODELS_CACHE_ROOT"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// Log parent-relative link or visual transforms under `entity_prefix`.
///
/// # Errors
///
/// Returns the Rerun recording error if a message cannot be logged.
pub fn log_transform_messages(
    recording: &rerun::RecordingStream,
    entity_prefix: &str,
    messages: &[RerunLinkTransform],
) -> Result<(), rerun::RecordingStreamError> {
    for message in messages {
        recording.log(
            prefixed_entity_path(entity_prefix, message.entity_path),
            &message.transform,
        )?;
    }
    Ok(())
}

/// Log mesh assets for visual entities.
///
/// Each mesh is read from `cache_root` joined with [`VisualAsset::mesh_resource`].
/// Log the matching visual-origin transforms separately so the mesh sits in the
/// link frame.
///
/// # Errors
///
/// Returns a read error when a mesh file is missing, or a Rerun error when a
/// mesh cannot be logged.
pub fn log_visual_meshes(
    recording: &rerun::RecordingStream,
    entity_prefix: &str,
    cache_root: &Path,
    visuals: &[VisualAsset],
) -> Result<(), TransformLogError> {
    for visual in visuals {
        let path = cache_root.join(visual.mesh_resource);
        let asset = rerun::Asset3D::from_file_path(&path).map_err(|source| {
            TransformLogError::ReadMesh {
                path: path.clone(),
                source,
            }
        })?;
        recording.log_static(
            prefixed_entity_path(entity_prefix, visual.entity_path),
            &asset,
        )?;
    }
    Ok(())
}

/// Whether two rotations represent the same orientation, including quaternion sign.
#[must_use]
pub fn rotations_match(actual: UnitQuaternion<f64>, expected: UnitQuaternion<f64>) -> bool {
    actual.coords.dot(&expected.coords).abs() > 1.0 - 1e-8
}

fn rotation_about(axis: [f64; 3], angle: f64) -> Isometry3<f64> {
    Isometry3::from_parts(
        nalgebra::Translation3::identity(),
        UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(Vector3::new(axis[0], axis[1], axis[2])),
            angle,
        ),
    )
}

fn hardware_model_cache_root_from(
    override_root: Option<std::ffi::OsString>,
    xdg_cache_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> PathBuf {
    if let Some(root) = override_root {
        return PathBuf::from(root);
    }
    let mut base = xdg_cache_home.map_or_else(
        || {
            home.map_or_else(|| PathBuf::from("."), PathBuf::from)
                .join(".cache")
        },
        PathBuf::from,
    );
    base.push("openstompers");
    base
}

#[cfg(test)]
mod tests {
    use super::{
        Isometry3, hardware_model_cache_root_from, prefixed_entity_path, world_from_local,
    };
    use nalgebra::Vector3;

    #[test]
    fn cache_root_matches_the_hardware_model_convention() {
        assert_eq!(
            hardware_model_cache_root_from(Some("/tmp/models".into()), None, None),
            PathBuf::from("/tmp/models")
        );
        assert_eq!(
            hardware_model_cache_root_from(None, Some("/xdg".into()), Some("/home/robot".into())),
            PathBuf::from("/xdg/openstompers")
        );
        assert_eq!(
            hardware_model_cache_root_from(None, None, Some("/home/robot".into())),
            PathBuf::from("/home/robot/.cache/openstompers")
        );
    }

    #[test]
    fn entity_prefix_is_optional() {
        assert_eq!(prefixed_entity_path("", "base_link"), "base_link");
        assert_eq!(
            prefixed_entity_path("robot", "base_link/visual_0"),
            "robot/base_link/visual_0"
        );
    }

    #[test]
    fn world_transforms_compose_in_parent_order() {
        let local = [
            Isometry3::identity(),
            Isometry3::translation(1.0, 0.0, 0.0),
            Isometry3::translation(0.0, 2.0, 0.0),
        ];
        let world = world_from_local(&local, &[0, 0, 1]);
        assert!((world[2].translation.vector - Vector3::new(1.0, 2.0, 0.0)).norm() < 1e-12);
    }

    use std::path::PathBuf;
}
