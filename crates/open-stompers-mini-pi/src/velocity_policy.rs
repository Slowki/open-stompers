//! Reference-free ONNX velocity control for all 23 Mini Pi Plus joints.

use std::path::Path;

use ort::{Session, Tensor};
use serde::Deserialize;
use thiserror::Error;

/// Number of named joints in the supported Mini Pi Plus variant.
pub const JOINT_COUNT: usize = 23;

/// Physical sensor and motor-target contract saved beside an exported actor.
#[derive(Clone, Debug, Deserialize)]
pub struct ObservationContract {
    pub hardware: String,
    pub joint_names: Vec<String>,
    pub home: Vec<f32>,
    pub lower: Vec<f32>,
    pub upper: Vec<f32>,
    pub velocity_limits: Vec<f32>,
    pub effort_limits: Vec<f32>,
    pub proportional_gains: Vec<f32>,
    pub derivative_gains: Vec<f32>,
    pub action_scale: f32,
    pub control_interval: f32,
    pub physics_interval: f32,
    #[serde(default = "default_command_rate_limit")]
    pub command_rate_limit: f32,
}

fn default_command_rate_limit() -> f32 {
    0.8
}

#[derive(Deserialize)]
struct ActorSettings {
    history: usize,
}

#[derive(Deserialize)]
struct ActorMetadata {
    kind: String,
    actor: ActorSettings,
    contract: ObservationContract,
    action_history: String,
}

/// Base-frame sensors in the metadata's named joint ordering.
#[derive(Clone, Debug, Deserialize)]
pub struct VelocityObservation {
    /// Gyroscope measurements in radians per second.
    pub angular_velocity: [f32; 3],
    /// Unit gravity vector projected into the base frame.
    pub projected_gravity: [f32; 3],
    /// Joint angles in radians.
    pub joint_positions: [f32; JOINT_COUNT],
    /// Joint angular velocities in radians per second.
    pub joint_velocities: [f32; JOINT_COUNT],
    /// Forward/lateral meters per second and yaw radians per second.
    pub desired_velocity: [f32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use uom::si::angle::radian;
    use uom::si::angular_velocity::radian_per_second;
    use uom::si::f32::{Angle, AngularVelocity};

    #[derive(Deserialize)]
    struct InferenceFixture {
        observation: VelocityObservation,
        policy_actions: [f32; JOINT_COUNT],
        joint_targets: [f32; JOINT_COUNT],
    }

    #[test]
    #[ignore = "requires OPEN_STOMPERS_VELOCITY_ACTOR and its exported runtime fixture"]
    fn exported_actor_matches_python_and_reset() {
        let path = std::path::PathBuf::from(
            std::env::var("OPEN_STOMPERS_VELOCITY_ACTOR").expect("exported actor path"),
        );
        let fixtures: Vec<InferenceFixture> = serde_json::from_slice(
            &std::fs::read(path.with_file_name("runtime_fixture.json"))
                .expect("Python inference fixture"),
        )
        .expect("valid fixture");
        let mut actor = VelocityPolicy::from_model_path(&path).expect("valid exported actor");
        for fixture in &fixtures {
            let observation = &fixture.observation;
            let angular_velocity =
                AngularVelocity::new::<radian_per_second>(observation.angular_velocity[0]);
            assert!(angular_velocity.get::<radian_per_second>().is_finite());
            let output = actor.infer(observation).expect("finite actor output");
            for index in 0..JOINT_COUNT {
                let target = Angle::new::<radian>(output.joint_targets[index]);
                assert!((target.get::<radian>() - fixture.joint_targets[index]).abs() < 1e-4);
                assert!(
                    (output.policy_actions[index] - fixture.policy_actions[index]).abs() < 1e-4
                );
            }
        }
        actor.reset();
        let output = actor
            .infer(&fixtures[0].observation)
            .expect("reset actor output");
        for index in 0..JOINT_COUNT {
            assert!(
                (output.policy_actions[index] - fixtures[0].policy_actions[index]).abs() < 1e-4
            );
        }
        let mut latencies = Vec::with_capacity(128);
        for index in 0..136 {
            let started = std::time::Instant::now();
            actor
                .infer(&fixtures[0].observation)
                .expect("timed actor output");
            if index >= 8 {
                latencies.push(started.elapsed());
            }
        }
        latencies.sort_unstable();
        let percentile = latencies[126].as_secs_f32();
        println!("Rust velocity adapter p99 seconds: {percentile}");
        assert!(percentile < actor.contract().control_interval);
    }
}

/// Deterministic policy actions and bounded position targets.
#[derive(Clone, Debug)]
pub struct VelocityOutput {
    pub policy_actions: [f32; JOINT_COUNT],
    pub joint_targets: [f32; JOINT_COUNT],
}

/// Actor loading, metadata, or inference failure.
#[derive(Debug, Error)]
pub enum VelocityPolicyError {
    #[error("ONNX Runtime failed: {0}")]
    Runtime(#[from] ort::Error),
    #[error("actor metadata could not be read: {0}")]
    Io(#[from] std::io::Error),
    #[error("actor metadata could not be decoded: {0}")]
    Metadata(#[from] serde_json::Error),
    #[error("actor contract does not match the 23-joint velocity interface")]
    InvalidContract,
    #[error("actor inputs or outputs contain nonfinite values")]
    NonFinite,
}

/// Stateful deterministic actor without reference motions or gait selectors.
pub struct VelocityPolicy {
    session: Session,
    contract: ObservationContract,
    history_length: usize,
    tokens: Vec<f32>,
    valid: Vec<bool>,
    previous_action: [f32; JOINT_COUNT],
    conditioned_velocity: [f32; 3],
}

impl ObservationContract {
    /// Validate physical metadata before allocating actor state.
    fn validate(&self) -> Result<(), VelocityPolicyError> {
        let groups = [
            &self.home,
            &self.lower,
            &self.upper,
            &self.velocity_limits,
            &self.effort_limits,
            &self.proportional_gains,
            &self.derivative_gains,
        ];
        let unique: std::collections::HashSet<_> = self.joint_names.iter().collect();
        if self.hardware != "PiPlus_S_12L8A0G2H1W"
            || self.joint_names.len() != JOINT_COUNT
            || unique.len() != JOINT_COUNT
            || groups.iter().any(|values| {
                values.len() != JOINT_COUNT || values.iter().any(|value| !value.is_finite())
            })
            || !self.action_scale.is_finite()
            || self.action_scale <= 0.0
            || !self.control_interval.is_finite()
            || (self.control_interval - 0.02).abs() > 1e-6
            || !self.physics_interval.is_finite()
            || self.physics_interval <= 0.0
            || !self.command_rate_limit.is_finite()
            || self.command_rate_limit <= 0.0
            || (0..JOINT_COUNT).any(|index| {
                self.velocity_limits[index] <= 0.0
                    || self.effort_limits[index] <= 0.0
                    || self.lower[index] >= self.upper[index]
                    || self.home[index] < self.lower[index]
                    || self.home[index] > self.upper[index]
            })
        {
            return Err(VelocityPolicyError::InvalidContract);
        }
        Ok(())
    }
}

impl VelocityPolicy {
    /// Load an actor and its adjacent JSON contract.
    pub fn from_model_path(path: &Path) -> Result<Self, VelocityPolicyError> {
        let metadata: ActorMetadata =
            serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?;
        metadata.contract.validate()?;
        if metadata.kind != "velocity_locomotion_v1"
            || metadata.action_history != "sampled_policy_action_before_reference_prior"
            || metadata.actor.history == 0
            || metadata.actor.history > 100
        {
            return Err(VelocityPolicyError::InvalidContract);
        }
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(path)?;
        if session.inputs.len() != 3
            || session.outputs.len() != 2
            || session.inputs[0].name != "history_tokens"
            || session.inputs[1].name != "history_valid"
            || session.inputs[2].name != "desired_velocity"
            || session.outputs[0].name != "policy_actions"
            || session.outputs[1].name != "joint_targets"
        {
            return Err(VelocityPolicyError::InvalidContract);
        }
        let history_length = metadata.actor.history;
        Ok(Self {
            session,
            contract: metadata.contract,
            history_length,
            tokens: vec![0.0; history_length * (6 + 3 * JOINT_COUNT)],
            valid: vec![false; history_length],
            previous_action: [0.0; JOINT_COUNT],
            conditioned_velocity: [0.0; 3],
        })
    }

    /// Return named joint ordering, limits, gains, and timing.
    pub fn contract(&self) -> &ObservationContract {
        &self.contract
    }

    /// Clear sensor history and previous sampled actions.
    pub fn reset(&mut self) {
        self.tokens.fill(0.0);
        self.valid.fill(false);
        self.previous_action.fill(0.0);
        self.conditioned_velocity.fill(0.0);
    }

    /// Infer bounded joint-position targets from sensors and desired base velocity.
    pub fn infer(
        &mut self,
        observation: &VelocityObservation,
    ) -> Result<VelocityOutput, VelocityPolicyError> {
        if observation
            .angular_velocity
            .iter()
            .chain(&observation.projected_gravity)
            .chain(&observation.joint_positions)
            .chain(&observation.joint_velocities)
            .chain(&observation.desired_velocity)
            .any(|value| !value.is_finite())
        {
            return Err(VelocityPolicyError::NonFinite);
        }
        let frame_width = 6 + 3 * JOINT_COUNT;
        self.tokens.rotate_left(frame_width);
        self.valid.rotate_left(1);
        self.valid[self.history_length - 1] = true;
        let frame = &mut self.tokens[(self.history_length - 1) * frame_width..];
        frame[..3].copy_from_slice(&observation.angular_velocity);
        frame[3..6].copy_from_slice(&observation.projected_gravity);
        for index in 0..JOINT_COUNT {
            frame[6 + index] = observation.joint_positions[index] - self.contract.home[index];
            frame[6 + JOINT_COUNT + index] =
                observation.joint_velocities[index] / self.contract.velocity_limits[index];
            frame[6 + 2 * JOINT_COUNT + index] = self.previous_action[index];
        }
        let tokens =
            Tensor::from_array(([1, self.history_length, frame_width], self.tokens.clone()))?;
        let valid = Tensor::from_array(([1, self.history_length], self.valid.clone()))?;
        let maximum_change = self.contract.command_rate_limit * self.contract.control_interval;
        for index in 0..3 {
            self.conditioned_velocity[index] += (observation.desired_velocity[index]
                - self.conditioned_velocity[index])
                .clamp(-maximum_change, maximum_change);
        }
        let velocity = Tensor::from_array(([1, 3], self.conditioned_velocity.to_vec()))?;
        let outputs = self.session.run(ort::inputs!["history_tokens" => tokens,
            "history_valid" => valid, "desired_velocity" => velocity]?)?;
        let (_, actions) = outputs[0].try_extract_raw_tensor::<f32>()?;
        let (_, targets) = outputs[1].try_extract_raw_tensor::<f32>()?;
        if actions.len() != JOINT_COUNT || targets.len() != JOINT_COUNT {
            return Err(VelocityPolicyError::InvalidContract);
        }
        if actions
            .iter()
            .chain(targets)
            .any(|value| !value.is_finite())
        {
            return Err(VelocityPolicyError::NonFinite);
        }
        let mut output = VelocityOutput {
            policy_actions: [0.0; JOINT_COUNT],
            joint_targets: [0.0; JOINT_COUNT],
        };
        output.policy_actions.copy_from_slice(actions);
        for (index, target) in targets.iter().enumerate() {
            output.joint_targets[index] =
                target.clamp(self.contract.lower[index], self.contract.upper[index]);
        }
        self.previous_action = output.policy_actions;
        Ok(output)
    }
}
