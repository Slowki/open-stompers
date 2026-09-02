//! Rerun visualization of a connected robot model.
//!
//! [`RobotModel`] is the platform-independent handle. Each variant owns that
//! platform's joint positions and logs meshes and link transforms under the
//! same entity path.

use open_stompers_hardware::transforms::{TransformLogError, hardware_model_cache_root};
use open_stompers_microduck::{
    JOINT_DEFINITIONS as MICRODUCK_JOINTS,
    transforms::microduck::{self as microduck, JointPositions as MicroduckJoints},
};
use open_stompers_mini_pi::{
    JOINT_DEFINITIONS as MINI_PI_JOINTS,
    transforms::pi_plus_s_12l8a0g2h1w::{self as mini_pi_plus, JointPositions as MiniPiPlusJoints},
};
use rerun::RecordingStream;
use rerun::blueprint::{Blueprint, Spatial3DView, Tabs, TimeSeriesView, Vertical};

use crate::config::Platform;

/// Entity path under which every robot kinematic tree is logged.
const ROBOT_ENTITY: &str = "robot";

/// Default Rerun layout for the selected robot platform.
#[must_use]
pub fn default_blueprint(platform: Platform) -> Blueprint {
    let (position_paths, torque_paths) = match platform {
        Platform::MiniPiPlus => {
            let attached_joints = MINI_PI_JOINTS
                .iter()
                .filter(|joint| joint.attached)
                .map(|joint| joint.name);
            motor_chart_paths(attached_joints)
        }
        Platform::Microduck => motor_chart_paths(
            MICRODUCK_JOINTS
                .iter()
                .filter(|joint| joint.name != "mouth")
                .map(|joint| joint.name),
        ),
    };

    let robot_view = Spatial3DView::new("3D View")
        .with_origin(ROBOT_ENTITY)
        .with_contents(["$origin/**"]);
    let position_view = TimeSeriesView::new("Positions (radians)")
        .with_origin("motors")
        .with_contents(position_paths);
    let torque_view = TimeSeriesView::new("Torques (newton meters)")
        .with_origin("motors")
        .with_contents(torque_paths);
    let policy_view = TimeSeriesView::new("Latency and inference time (seconds)")
        .with_origin("policy")
        .with_contents([
            "policy/motor_data_to_command_latency_seconds",
            "policy/inference_duration_seconds",
        ]);
    let policy_tab = Vertical::new([policy_view.into()]).with_name("Policy");
    let motors_view = Vertical::new([position_view.into(), torque_view.into()])
        .with_name("Motors")
        .with_row_shares([1.0, 1.0]);

    Blueprint::new(
        Tabs::new([robot_view.into(), motors_view.into(), policy_tab.into()])
            .with_name("Open Stompers"),
    )
    .with_auto_views(false)
    .with_auto_layout(false)
}

fn motor_chart_paths<'a>(
    joint_names: impl IntoIterator<Item = &'a str>,
) -> (Vec<String>, Vec<String>) {
    let mut position_paths = Vec::new();
    let mut torque_paths = Vec::new();
    for joint_name in joint_names {
        position_paths.push(format!("motors/{joint_name}/position_rad"));
        torque_paths.push(format!("motors/{joint_name}/torque_newton_meters"));
    }
    (position_paths, torque_paths)
}

/// Joint positions and visual assets for the connected platform.
#[derive(Clone, Debug)]
pub enum RobotModel {
    /// Mini Pi Plus variant `PiPlus_S_12L8A0G2H1W`.
    MiniPiPlus(MiniPiPlusJoints),
    /// Microduck kinematic model.
    Microduck(MicroduckJoints),
}

impl RobotModel {
    /// Empty joint positions for `platform`.
    #[must_use]
    pub fn new(platform: Platform) -> Self {
        match platform {
            Platform::MiniPiPlus => Self::mini_pi_plus(),
            Platform::Microduck => Self::microduck(),
        }
    }

    /// Empty Mini Pi Plus joint positions.
    #[must_use]
    pub fn mini_pi_plus() -> Self {
        Self::MiniPiPlus(MiniPiPlusJoints::default())
    }

    /// Empty Microduck joint positions.
    #[must_use]
    pub fn microduck() -> Self {
        Self::Microduck(MicroduckJoints::default())
    }

    /// Log static visual meshes and their link-frame origins.
    ///
    /// # Errors
    ///
    /// Returns a mesh read error or a Rerun logging error.
    pub fn log_meshes(&self, recording: &RecordingStream) -> Result<(), TransformLogError> {
        let cache_root = hardware_model_cache_root();
        match self {
            Self::MiniPiPlus(_) => {
                mini_pi_plus::log_rerun_visual_transforms(recording, ROBOT_ENTITY)?;
                mini_pi_plus::log_rerun_visual_meshes(recording, ROBOT_ENTITY, &cache_root)
            }
            Self::Microduck(_) => {
                microduck::log_rerun_visual_transforms(recording, ROBOT_ENTITY)?;
                microduck::log_rerun_visual_meshes(recording, ROBOT_ENTITY, &cache_root)
            }
        }
    }

    /// Log parent-relative link transforms for the current joint positions.
    ///
    /// # Errors
    ///
    /// Returns the Rerun recording error if a transform cannot be logged.
    pub fn log_transforms(
        &self,
        recording: &RecordingStream,
    ) -> Result<(), rerun::RecordingStreamError> {
        match self {
            Self::MiniPiPlus(positions) => {
                mini_pi_plus::log_rerun_link_transforms(recording, ROBOT_ENTITY, positions)
            }
            Self::Microduck(positions) => {
                microduck::log_rerun_link_transforms(recording, ROBOT_ENTITY, positions)
            }
        }
    }

    /// Update one joint from its URDF name.
    ///
    /// Returns `false` when `joint_name` is not an independent joint of this
    /// model. Positions are radians.
    pub fn record_joint(&mut self, joint_name: &str, radians: f64) -> bool {
        let recorded = match self {
            Self::MiniPiPlus(positions) => positions.set_named(joint_name, radians),
            Self::Microduck(positions) => positions.set_named(joint_name, radians),
        };
        if !recorded {
            tracing::debug!("joint {joint_name} is not part of the visualized robot model");
        }
        recorded
    }
}

#[cfg(test)]
mod tests {
    use open_stompers_microduck::JOINT_DEFINITIONS as MICRODUCK_JOINTS;
    use open_stompers_mini_pi::JOINT_DEFINITIONS as MINI_PI_JOINTS;

    use super::{RobotModel, default_blueprint, motor_chart_paths};
    use crate::config::Platform;

    #[test]
    fn motor_chart_paths_separate_positions_and_torques() {
        let (positions, torques) = motor_chart_paths(["left_knee", "right_knee"]);

        assert_eq!(
            positions,
            [
                "motors/left_knee/position_rad",
                "motors/right_knee/position_rad"
            ]
        );
        assert_eq!(
            torques,
            [
                "motors/left_knee/torque_newton_meters",
                "motors/right_knee/torque_newton_meters"
            ]
        );
    }

    #[test]
    fn default_blueprints_are_written_for_each_platform() {
        for platform in [Platform::MiniPiPlus, Platform::Microduck] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("blueprint.rrd");
            let recording = rerun::RecordingStreamBuilder::new("open-stompers-blueprint-test")
                .with_default_blueprint(default_blueprint(platform))
                .save(&path)
                .unwrap();
            recording.flush_blocking().unwrap();

            assert!(std::fs::metadata(path).unwrap().len() > 0);
        }
    }

    #[test]
    fn attached_mini_pi_motors_exist_in_the_visualized_model() {
        let mut model = RobotModel::mini_pi_plus();
        for joint in MINI_PI_JOINTS.iter().filter(|joint| joint.attached) {
            assert!(
                model.record_joint(joint.name, 0.2),
                "{} is missing from the visualized model",
                joint.name
            );
        }
    }

    #[test]
    fn microduck_motors_exist_in_the_visualized_model_except_the_mouth() {
        let mut model = RobotModel::microduck();
        for joint in MICRODUCK_JOINTS {
            assert_eq!(
                model.record_joint(joint.name, 0.2),
                joint.name != "mouth",
                "{}",
                joint.name
            );
        }
    }
}
