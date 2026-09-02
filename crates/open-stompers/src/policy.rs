//! Hardware-facing adapter for the velocity actor's named 23-joint contract.

use std::path::Path;

use open_stompers_mini_pi::JOINT_DEFINITIONS;
use open_stompers_mini_pi::transforms::pi_plus_s_12l8a0g2h1w::{JOINT_LIMITS, JOINT_NAMES};
use open_stompers_mini_pi::velocity_policy::{
    JOINT_COUNT, ObservationContract, VelocityObservation, VelocityOutput, VelocityPolicy,
    VelocityPolicyError,
};
use uom::si::angle::radian;
use uom::si::f64::Angle;

use crate::ApplicationError;

/// Named physical position target with the actor's controller gains.
#[derive(Clone, Copy)]
pub(super) struct MotorTarget {
    pub joint_name: &'static str,
    pub position: Angle,
    pub proportional_gain: f32,
    pub derivative_gain: f32,
}

/// Sampled actor actions and all 23 physical motor targets.
pub(super) struct DeploymentOutput {
    pub policy_actions: [f32; JOINT_COUNT],
    pub commands: [MotorTarget; JOINT_COUNT],
}

/// Deterministic velocity inference mapped to attached hardware joints.
pub(super) struct DeploymentPolicy {
    actor: VelocityPolicy,
    joint_names: [&'static str; JOINT_COUNT],
}

impl DeploymentPolicy {
    /// Load the actor and resolve every metadata joint before opening hardware.
    pub fn from_model_path(path: &Path) -> Result<Self, ApplicationError> {
        let actor = VelocityPolicy::from_model_path(path)?;
        validate_model_limits(actor.contract())?;
        let mut joint_names = [""; JOINT_COUNT];
        for (index, name) in actor.contract().joint_names.iter().enumerate() {
            joint_names[index] = JOINT_DEFINITIONS
                .iter()
                .find(|joint| joint.attached && joint.name == name)
                .ok_or_else(|| ApplicationError::MissingPolicyMotor {
                    joint_name: name.clone(),
                })?
                .name;
        }
        Ok(Self { actor, joint_names })
    }

    /// Return the exact sensor and controller contract.
    pub fn contract(&self) -> &ObservationContract {
        self.actor.contract()
    }

    /// Return joint names in actor observation and output order.
    pub fn joint_names(&self) -> &[&'static str; JOINT_COUNT] {
        &self.joint_names
    }

    /// Clear actor history after missing or stale telemetry.
    pub fn reset(&mut self) {
        self.actor.reset();
    }

    /// Infer all named motor targets without extra action filters or head overrides.
    pub fn infer(
        &mut self,
        observation: &VelocityObservation,
    ) -> Result<DeploymentOutput, VelocityPolicyError> {
        let output = self.actor.infer(observation)?;
        Ok(build_output(
            self.actor.contract(),
            &self.joint_names,
            output,
        ))
    }
}

fn validate_model_limits(contract: &ObservationContract) -> Result<(), VelocityPolicyError> {
    for (index, name) in contract.joint_names.iter().enumerate() {
        let model_index = JOINT_NAMES
            .iter()
            .position(|joint_name| joint_name == name)
            .ok_or(VelocityPolicyError::InvalidContract)?;
        let limits = JOINT_LIMITS[model_index].ok_or(VelocityPolicyError::InvalidContract)?;
        if f64::from(contract.lower[index]) < limits.minimum_position.get::<radian>() - 1e-5
            || f64::from(contract.upper[index]) > limits.maximum_position.get::<radian>() + 1e-5
        {
            return Err(VelocityPolicyError::InvalidContract);
        }
    }
    Ok(())
}

fn build_output(
    contract: &ObservationContract,
    joint_names: &[&'static str; JOINT_COUNT],
    output: VelocityOutput,
) -> DeploymentOutput {
    DeploymentOutput {
        policy_actions: output.policy_actions,
        commands: std::array::from_fn(|index| MotorTarget {
            joint_name: joint_names[index],
            position: Angle::new::<radian>(f64::from(output.joint_targets[index])),
            proportional_gain: contract.proportional_gains[index],
            derivative_gain: contract.derivative_gains[index],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires OPEN_STOMPERS_VELOCITY_ACTOR"]
    fn exported_actor_resolves_every_attached_motor_and_physical_limit() {
        let path = std::path::PathBuf::from(
            std::env::var("OPEN_STOMPERS_VELOCITY_ACTOR").expect("exported actor path"),
        );
        let policy = DeploymentPolicy::from_model_path(&path).expect("physical actor contract");
        assert_eq!(policy.joint_names().len(), JOINT_COUNT);
        assert!(policy.joint_names().contains(&"waist_yaw_joint"));
        assert!(policy.joint_names().contains(&"head_yaw_joint"));
        assert!(policy.joint_names().contains(&"head_pitch_joint"));
    }

    #[test]
    fn every_target_uses_metadata_order_and_gains() {
        let contract = ObservationContract {
            hardware: "PiPlus_S_12L8A0G2H1W".to_owned(),
            joint_names: JOINT_NAMES.iter().map(ToString::to_string).collect(),
            home: vec![0.0; JOINT_COUNT],
            lower: vec![-2.0; JOINT_COUNT],
            upper: vec![2.0; JOINT_COUNT],
            velocity_limits: vec![5.45; JOINT_COUNT],
            effort_limits: vec![20.0; JOINT_COUNT],
            proportional_gains: (0..JOINT_COUNT).map(|index| 30.0 + index as f32).collect(),
            derivative_gains: vec![2.0; JOINT_COUNT],
            action_scale: 0.35,
            control_interval: 0.02,
            physics_interval: 0.005,
            command_rate_limit: 0.8,
        };
        let output = build_output(
            &contract,
            &JOINT_NAMES,
            VelocityOutput {
                policy_actions: [0.5; JOINT_COUNT],
                joint_targets: [0.2; JOINT_COUNT],
            },
        );
        for (index, target) in output.commands.iter().enumerate() {
            assert_eq!(target.joint_name, JOINT_NAMES[index]);
            assert_eq!(target.proportional_gain, contract.proportional_gains[index]);
            assert_eq!(target.derivative_gain, 2.0);
            assert!((target.position.get::<radian>() - 0.2).abs() < 1e-6);
        }
    }
}
