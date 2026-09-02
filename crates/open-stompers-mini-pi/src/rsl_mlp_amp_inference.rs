//! CPU inference and action processing for the Mini Pi Plus RSL AMP actor.

use std::array;
use std::path::Path;

use ort::{Session, Tensor, TensorElementType, ValueType};
use thiserror::Error;
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{Angle, AngularVelocity};

use crate::mini_pi_plus::{
    HEAD_POSITION_DERIVATIVE_GAIN, HEAD_POSITION_PROPORTIONAL_GAIN, WAIST_POSITION_DERIVATIVE_GAIN,
    WAIST_POSITION_PROPORTIONAL_GAIN,
};
use open_stompers_hardware::GamepadCommand;

/// Number of joints controlled by the actor.
pub const POLICY_JOINT_COUNT: usize = 20;
/// Number of features in one policy observation frame.
pub const POLICY_FRAME_SIZE: usize = 69;
/// Number of observation frames supplied to the actor.
pub const POLICY_HISTORY_LENGTH: usize = 5;
/// Width of the actor input tensor.
pub const POLICY_INPUT_SIZE: usize = POLICY_FRAME_SIZE * POLICY_HISTORY_LENGTH;

/// Joint names in actor output order.
pub const POLICY_JOINT_NAMES: [&str; POLICY_JOINT_COUNT] = [
    "l_hip_pitch_joint",
    "l_shoulder_pitch_joint",
    "r_hip_pitch_joint",
    "r_shoulder_pitch_joint",
    "l_hip_roll_joint",
    "l_shoulder_roll_joint",
    "r_hip_roll_joint",
    "r_shoulder_roll_joint",
    "l_thigh_joint",
    "l_upper_arm_joint",
    "r_thigh_joint",
    "r_upper_arm_joint",
    "l_calf_joint",
    "l_elbow_joint",
    "r_calf_joint",
    "r_elbow_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
];

const OBSERVATION_CLIP: f32 = 18.0;
const ACTION_CLIP: f32 = std::f32::consts::PI;
const ACTION_FILTER_SCALE: f32 = 0.8;
const ACTION_SCALE: f32 = 0.25;
const WARMUP_CYCLES: usize = 10;
const COMMAND_LINEAR_VELOCITY_X_MINIMUM_METERS_PER_SECOND: f32 = -0.80;
const COMMAND_LINEAR_VELOCITY_X_MAXIMUM_METERS_PER_SECOND: f32 = 0.80;
const COMMAND_LINEAR_VELOCITY_Y_MINIMUM_METERS_PER_SECOND: f32 = -0.50;
const COMMAND_LINEAR_VELOCITY_Y_MAXIMUM_METERS_PER_SECOND: f32 = 0.50;
const COMMAND_ANGULAR_VELOCITY_Z_MINIMUM_RADIANS_PER_SECOND: f32 = -1.57;
const COMMAND_ANGULAR_VELOCITY_Z_MAXIMUM_RADIANS_PER_SECOND: f32 = 1.57;

/// Actor-order neutral joint positions.
pub const NEUTRAL_POSITIONS: [Angle; POLICY_JOINT_COUNT] = [
    open_stompers_hardware::angle!(-0.25),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(-0.25),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.2),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(-0.2),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.65),
    open_stompers_hardware::angle!(-1.2),
    open_stompers_hardware::angle!(0.65),
    open_stompers_hardware::angle!(-1.2),
    open_stompers_hardware::angle!(-0.4),
    open_stompers_hardware::angle!(-0.4),
    open_stompers_hardware::angle!(0.0),
    open_stompers_hardware::angle!(0.0),
];

/// Actor-order proportional gains from `pd.yaml`.
pub const POLICY_KP: [f32; POLICY_JOINT_COUNT] = [
    80.0, 30.0, 80.0, 30.0, 80.0, 30.0, 80.0, 30.0, 80.0, 30.0, 80.0, 30.0, 80.0, 30.0, 80.0, 30.0,
    60.0, 60.0, 60.0, 60.0,
];

/// Actor-order derivative gains from `pd.yaml`.
pub const POLICY_KD: [f32; POLICY_JOINT_COUNT] = [
    1.1, 0.6, 1.1, 0.6, 1.1, 0.6, 1.1, 0.6, 1.1, 0.6, 1.1, 0.6, 1.1, 0.6, 1.1, 0.6, 1.2, 1.2, 1.2,
    1.2,
];

/// Scale normalized gamepad axes to Mini Pi Plus AMP policy command limits.
pub fn scale_gamepad_input(command: GamepadCommand) -> [f32; 3] {
    let forward_axis = normalize_gamepad_axis(command.forward_axis);
    let lateral_axis = normalize_gamepad_axis(command.lateral_axis);
    let yaw_axis = normalize_gamepad_axis(command.yaw_axis);

    [
        scale_asymmetric_axis(
            forward_axis,
            COMMAND_LINEAR_VELOCITY_X_MINIMUM_METERS_PER_SECOND,
            COMMAND_LINEAR_VELOCITY_X_MAXIMUM_METERS_PER_SECOND,
        ),
        scale_asymmetric_axis(
            lateral_axis,
            COMMAND_LINEAR_VELOCITY_Y_MINIMUM_METERS_PER_SECOND,
            COMMAND_LINEAR_VELOCITY_Y_MAXIMUM_METERS_PER_SECOND,
        ),
        scale_asymmetric_axis(
            yaw_axis,
            COMMAND_ANGULAR_VELOCITY_Z_MINIMUM_RADIANS_PER_SECOND,
            COMMAND_ANGULAR_VELOCITY_Z_MAXIMUM_RADIANS_PER_SECOND,
        ),
    ]
}

fn normalize_gamepad_axis(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn scale_asymmetric_axis(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value < 0.0 {
        value * -minimum
    } else {
        value * maximum
    }
}

/// One measured policy input sample in actor joint order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyObservation {
    /// Base angular velocity in radians per second.
    pub angular_velocity: [AngularVelocity; 3],
    /// Gravity projected into the robot frame.
    pub projected_gravity: [f32; 3],
    /// Linear X, linear Y, and angular Z command in the policy's native units.
    pub command_velocity: [f32; 3],
    /// Joint positions in radians, in [`POLICY_JOINT_NAMES`] order.
    pub joint_positions: [Angle; POLICY_JOINT_COUNT],
    /// Joint velocities in radians per second, in actor order.
    pub joint_velocities: [AngularVelocity; POLICY_JOINT_COUNT],
}

/// Position target and gains for one attached joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPositionCommand {
    /// URDF joint name.
    pub joint_name: &'static str,
    /// Target position in radians.
    pub position: Angle,
    /// Position proportional gain.
    pub kp: f32,
    /// Position derivative gain.
    pub kd: f32,
}

/// Policy result suitable for logging or a later motor-control stage.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyOutput {
    /// Unclipped deterministic actor output.
    pub raw_actions: [f32; POLICY_JOINT_COUNT],
    /// Actions clipped to the deployment limits.
    pub clipped_actions: [f32; POLICY_JOINT_COUNT],
    /// Policy commands for all 23 attached joints.
    pub commands: [JointPositionCommand; 23],
}

/// ONNX Runtime or policy-input failure.
#[derive(Debug, Error)]
pub enum PolicyError {
    /// ONNX Runtime failed to initialize or infer.
    #[error("ONNX Runtime failed: {0}")]
    Onnx(#[from] ort::Error),
    /// The model does not expose the expected names and dimensions.
    #[error("model must have one `observations` input and one `actions` output")]
    InvalidModelInterface,
    /// Input or actor output contains a nonfinite value.
    #[error("policy observation or actor output contains a nonfinite value")]
    NonFiniteValue,
    /// The actor returned a different number of actions than expected.
    #[error("actor output must contain exactly {POLICY_JOINT_COUNT} values")]
    InvalidActionShape,
}

/// Stateful RSL AMP actor using CPU ONNX Runtime inference.
pub struct RslMlpAmpPolicy {
    session: Session,
    state: PolicyState,
}

/// Observation history and action memory for the deployment controller.
struct PolicyState {
    history: [[f32; POLICY_FRAME_SIZE]; POLICY_HISTORY_LENGTH],
    // Store clipped raw actions, never filtered actions or position targets.
    previous_actions: [f32; POLICY_JOINT_COUNT],
    warmup_remaining: usize,
}

impl Default for PolicyState {
    fn default() -> Self {
        Self {
            history: [[0.0; POLICY_FRAME_SIZE]; POLICY_HISTORY_LENGTH],
            previous_actions: [0.0; POLICY_JOINT_COUNT],
            warmup_remaining: WARMUP_CYCLES,
        }
    }
}

impl PolicyState {
    fn push_observation(
        &mut self,
        observation: &PolicyObservation,
    ) -> Result<Vec<f32>, PolicyError> {
        let frame = build_observation_frame(observation, &self.previous_actions)?;
        self.history.rotate_left(1);
        self.history[POLICY_HISTORY_LENGTH - 1] = frame;
        Ok(self.history.iter().flatten().copied().collect())
    }

    fn process_actions(&mut self, raw_actions: [f32; POLICY_JOINT_COUNT]) -> PolicyOutput {
        let output = build_policy_output(
            raw_actions,
            &self.previous_actions,
            self.warmup_remaining > 0,
        );
        self.previous_actions = output.clipped_actions;
        self.warmup_remaining = self.warmup_remaining.saturating_sub(1);
        output
    }
}

impl RslMlpAmpPolicy {
    /// Load and validate an ONNX actor from `model_path`.
    pub fn from_model_path(model_path: &Path) -> Result<Self, PolicyError> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        if session.inputs.len() != 1
            || session.outputs.len() != 1
            || session.inputs[0].name != "observations"
            || session.outputs[0].name != "actions"
            || !valid_tensor_type(&session.inputs[0].input_type, POLICY_INPUT_SIZE)
            || !valid_tensor_type(&session.outputs[0].output_type, POLICY_JOINT_COUNT)
        {
            return Err(PolicyError::InvalidModelInterface);
        }
        Ok(Self {
            session,
            state: PolicyState::default(),
        })
    }

    /// Clear policy memory and restart the neutral-pose warmup.
    pub fn reset(&mut self) {
        self.state = PolicyState::default();
    }

    /// Build one clipped observation frame in the actor's declared feature order.
    pub fn build_frame(
        &self,
        observation: &PolicyObservation,
    ) -> Result<[f32; POLICY_FRAME_SIZE], PolicyError> {
        build_observation_frame(observation, &self.state.previous_actions)
    }

    /// Run the actor and prepare filtered targets after the neutral-pose warmup.
    pub fn infer(&mut self, observation: &PolicyObservation) -> Result<PolicyOutput, PolicyError> {
        let input_values = self.state.push_observation(observation)?;
        let input = Tensor::from_array(([1, POLICY_INPUT_SIZE], input_values))?;
        let outputs = self.session.run([input.into()])?;
        let (shape, raw_view) = outputs[0].try_extract_raw_tensor::<f32>()?;
        if shape != [1, POLICY_JOINT_COUNT as i64] || raw_view.len() != POLICY_JOINT_COUNT {
            return Err(PolicyError::InvalidActionShape);
        }
        let raw_actions: [f32; POLICY_JOINT_COUNT] = raw_view
            .try_into()
            .map_err(|_| PolicyError::InvalidActionShape)?;
        if !raw_actions.iter().all(|value| value.is_finite()) {
            return Err(PolicyError::NonFiniteValue);
        }
        Ok(self.state.process_actions(raw_actions))
    }
}

#[allow(clippy::cast_possible_truncation)]
fn build_observation_frame(
    observation: &PolicyObservation,
    previous_actions: &[f32; POLICY_JOINT_COUNT],
) -> Result<[f32; POLICY_FRAME_SIZE], PolicyError> {
    let mut frame = [0.0; POLICY_FRAME_SIZE];
    for (slot, velocity) in frame[..3].iter_mut().zip(observation.angular_velocity) {
        *slot = velocity.get::<radian_per_second>() as f32;
    }
    frame[3..6].copy_from_slice(&observation.projected_gravity);
    frame[6..9].copy_from_slice(&observation.command_velocity);
    for (index, (slot, position)) in frame[9..29]
        .iter_mut()
        .zip(observation.joint_positions)
        .enumerate()
    {
        *slot = (position - NEUTRAL_POSITIONS[index]).get::<radian>() as f32;
    }
    for (slot, velocity) in frame[29..49].iter_mut().zip(observation.joint_velocities) {
        *slot = velocity.get::<radian_per_second>() as f32;
    }
    frame[49..69].copy_from_slice(previous_actions);
    if !frame.iter().all(|value| value.is_finite()) {
        return Err(PolicyError::NonFiniteValue);
    }
    frame
        .iter_mut()
        .for_each(|value| *value = value.clamp(-OBSERVATION_CLIP, OBSERVATION_CLIP));
    Ok(frame)
}

#[allow(clippy::cast_possible_truncation)]
fn build_policy_output(
    raw_actions: [f32; POLICY_JOINT_COUNT],
    previous_actions: &[f32; POLICY_JOINT_COUNT],
    warming_up: bool,
) -> PolicyOutput {
    let clipped_actions = raw_actions.map(|action| action.clamp(-ACTION_CLIP, ACTION_CLIP));
    let policy_commands: [JointPositionCommand; POLICY_JOINT_COUNT] = array::from_fn(|index| {
        // Use the previous clipped raw action, matching the reference's two-tap filter.
        let filtered_action = ACTION_FILTER_SCALE.mul_add(
            clipped_actions[index],
            (1.0 - ACTION_FILTER_SCALE) * previous_actions[index],
        );
        let position = if warming_up {
            NEUTRAL_POSITIONS[index]
        } else {
            Angle::new::<radian>(f64::from(ACTION_SCALE.mul_add(
                filtered_action,
                NEUTRAL_POSITIONS[index].get::<radian>() as f32,
            )))
        };
        JointPositionCommand {
            joint_name: POLICY_JOINT_NAMES[index],
            position,
            kp: POLICY_KP[index],
            kd: POLICY_KD[index],
        }
    });
    let extra_commands = [
        JointPositionCommand {
            joint_name: "head_yaw_joint",
            position: Angle::new::<radian>(0.0),
            kp: HEAD_POSITION_PROPORTIONAL_GAIN,
            kd: HEAD_POSITION_DERIVATIVE_GAIN,
        },
        JointPositionCommand {
            joint_name: "head_pitch_joint",
            position: Angle::new::<radian>(0.0),
            kp: HEAD_POSITION_PROPORTIONAL_GAIN,
            kd: HEAD_POSITION_DERIVATIVE_GAIN,
        },
        JointPositionCommand {
            joint_name: "waist_yaw_joint",
            position: Angle::new::<radian>(0.0),
            kp: WAIST_POSITION_PROPORTIONAL_GAIN,
            kd: WAIST_POSITION_DERIVATIVE_GAIN,
        },
    ];
    let commands = std::array::from_fn(|index| {
        if index < POLICY_JOINT_COUNT {
            policy_commands[index]
        } else {
            extra_commands[index - POLICY_JOINT_COUNT]
        }
    });
    PolicyOutput {
        raw_actions,
        clipped_actions,
        commands,
    }
}

fn valid_tensor_type(value_type: &ValueType, feature_width: usize) -> bool {
    matches!(
        value_type,
        ValueType::Tensor { ty: TensorElementType::Float32, dimensions }
            if dimensions.len() == 2
                && dimensions[0] <= 1
                && dimensions[0] != 0
                && dimensions[1] == feature_width as i64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> PolicyObservation {
        PolicyObservation {
            angular_velocity: [1.0, 2.0, 3.0].map(AngularVelocity::new::<radian_per_second>),
            projected_gravity: [0.0, 0.0, -1.0],
            command_velocity: [0.4, -0.5, 1.57],
            joint_positions: NEUTRAL_POSITIONS
                .map(|position| Angle::new::<radian>(position.get::<radian>() + 0.5)),
            joint_velocities: [4.0; POLICY_JOINT_COUNT]
                .map(AngularVelocity::new::<radian_per_second>),
        }
    }

    #[test]
    fn observation_frame_uses_policy_order_and_neutral_residuals() {
        let frame = build_observation_frame(&observation(), &[0.25; POLICY_JOINT_COUNT]).unwrap();
        assert_eq!(&frame[..6], &[1.0, 2.0, 3.0, 0.0, 0.0, -1.0]);
        assert_eq!(&frame[6..9], &[0.4, -0.5, 1.57]);
        assert_eq!(&frame[9..29], &[0.5; POLICY_JOINT_COUNT]);
        assert_eq!(&frame[29..49], &[4.0; POLICY_JOINT_COUNT]);
        assert_eq!(&frame[49..69], &[0.25; POLICY_JOINT_COUNT]);
    }

    #[test]
    fn frame_rejects_nonfinite_input_and_clips_finite_values() {
        let mut sample = observation();
        sample.joint_velocities[3] = AngularVelocity::new::<radian_per_second>(f64::INFINITY);
        assert!(matches!(
            build_observation_frame(&sample, &[0.0; POLICY_JOINT_COUNT]),
            Err(PolicyError::NonFiniteValue)
        ));
        let mut sample = observation();
        sample.angular_velocity[0] = AngularVelocity::new::<radian_per_second>(100.0);
        assert_eq!(
            build_observation_frame(&sample, &[0.0; POLICY_JOINT_COUNT]).unwrap()[0],
            OBSERVATION_CLIP
        );
    }

    #[test]
    fn postprocessing_clips_scales_and_maps_all_attached_joints() {
        let raw_actions = [20.0; POLICY_JOINT_COUNT];
        let output = build_policy_output(raw_actions, &[0.0; POLICY_JOINT_COUNT], false);
        assert_eq!(output.clipped_actions, [ACTION_CLIP; POLICY_JOINT_COUNT]);
        assert_eq!(output.commands.len(), 23);
        assert_eq!(output.commands[0].joint_name, "l_hip_pitch_joint");
        let expected_position = -0.25 + 0.25 * 0.8 * std::f64::consts::PI;
        assert!((output.commands[0].position.get::<radian>() - expected_position).abs() < 1e-6);
        assert_eq!(output.commands[0].kp, 80.0);
        assert_eq!(output.commands[0].kd, 1.1);
        assert_eq!(output.commands[20].joint_name, "head_yaw_joint");
        assert_eq!(output.commands[20].kp, HEAD_POSITION_PROPORTIONAL_GAIN);
        assert_eq!(output.commands[20].kd, HEAD_POSITION_DERIVATIVE_GAIN);
        assert_eq!(output.commands[21].joint_name, "head_pitch_joint");
        assert_eq!(output.commands[21].kp, HEAD_POSITION_PROPORTIONAL_GAIN);
        assert_eq!(output.commands[21].kd, HEAD_POSITION_DERIVATIVE_GAIN);
        assert_eq!(output.commands[22].joint_name, "waist_yaw_joint");
        assert_eq!(output.commands[22].kp, WAIST_POSITION_PROPORTIONAL_GAIN);
        assert_eq!(output.commands[22].kd, WAIST_POSITION_DERIVATIVE_GAIN);
        for (index, command) in output.commands[..POLICY_JOINT_COUNT].iter().enumerate() {
            assert_eq!(command.joint_name, POLICY_JOINT_NAMES[index]);
            let residual = (command.position - NEUTRAL_POSITIONS[index]).get::<radian>();
            assert!((residual - 0.25 * 0.8 * std::f64::consts::PI).abs() < 1e-6);
        }
    }

    #[test]
    fn action_filter_uses_previous_clipped_raw_action() {
        let mut state = PolicyState {
            warmup_remaining: 0,
            ..PolicyState::default()
        };
        // Saturation followed by a sign reversal distinguishes raw memory from a smoother.
        for (raw_action, expected_filtered) in [
            (20.0, 0.8 * std::f64::consts::PI),
            (-20.0, -0.6 * std::f64::consts::PI),
            (0.0, -0.2 * std::f64::consts::PI),
            (0.0, 0.0),
        ] {
            let output = state.process_actions([raw_action; POLICY_JOINT_COUNT]);
            let residual = (output.commands[0].position - NEUTRAL_POSITIONS[0]).get::<radian>();
            assert!((residual - 0.25 * expected_filtered).abs() < 1e-6);
            let frame = build_observation_frame(&observation(), &state.previous_actions).unwrap();
            assert_eq!(
                &frame[49..],
                &[raw_action.clamp(-ACTION_CLIP, ACTION_CLIP); POLICY_JOINT_COUNT]
            );
        }
    }

    #[test]
    fn warmup_holds_neutral_while_populating_history_and_action_memory() {
        let mut state = PolicyState::default();
        for cycle in 0..WARMUP_CYCLES {
            let mut sample = observation();
            sample.angular_velocity[0] = AngularVelocity::new::<radian_per_second>(cycle as f64);
            let input = state.push_observation(&sample).unwrap();
            for (frame_index, frame) in input.as_chunks::<POLICY_FRAME_SIZE>().0.iter().enumerate()
            {
                let age = POLICY_HISTORY_LENGTH - 1 - frame_index;
                if age > cycle {
                    assert_eq!(frame, &[0.0; POLICY_FRAME_SIZE]);
                } else {
                    let frame_cycle = cycle - age;
                    assert_eq!(frame[0], frame_cycle as f32);
                    assert_eq!(
                        &frame[49..],
                        &[if frame_cycle == 0 { 0.0 } else { 1.0 }; POLICY_JOINT_COUNT]
                    );
                }
            }
            let output = state.process_actions([1.0; POLICY_JOINT_COUNT]);
            for (command, neutral) in output.commands.iter().zip(NEUTRAL_POSITIONS) {
                assert_eq!(command.position, neutral);
            }
            assert_eq!(output.raw_actions, [1.0; POLICY_JOINT_COUNT]);
            assert_eq!(state.previous_actions, [1.0; POLICY_JOINT_COUNT]);
        }
        let output = state.process_actions([2.0; POLICY_JOINT_COUNT]);
        let residual = (output.commands[0].position - NEUTRAL_POSITIONS[0]).get::<radian>();
        assert!((residual - 0.25 * (0.8 * 2.0 + 0.2)).abs() < 1e-6);
    }

    #[test]
    fn tensor_interface_accepts_only_expected_float_width() {
        assert!(valid_tensor_type(
            &ValueType::Tensor {
                ty: TensorElementType::Float32,
                dimensions: vec![-1, 345]
            },
            345
        ));
        assert!(!valid_tensor_type(
            &ValueType::Tensor {
                ty: TensorElementType::Float32,
                dimensions: vec![1, 346]
            },
            345
        ));
        assert!(!valid_tensor_type(
            &ValueType::Tensor {
                ty: TensorElementType::Int64,
                dimensions: vec![1, 345]
            },
            345
        ));
    }

    #[test]
    fn loads_and_runs_configured_onnx_model() {
        let Some(model_path) = std::env::var_os("OPEN_STOMPERS_POLICY_MODEL") else {
            return;
        };
        let mut policy = RslMlpAmpPolicy::from_model_path(Path::new(&model_path)).unwrap();
        let output = policy.infer(&observation()).unwrap();
        assert!(output.raw_actions.iter().all(|value| value.is_finite()));
        assert!(
            output
                .commands
                .iter()
                .all(|command| command.position.get::<radian>().is_finite())
        );
        let expected = [
            0.10307045,
            0.22117174,
            -0.07277588,
            -0.2897556,
            -0.59257954,
            0.51928264,
            -0.06563663,
            -0.13419387,
            0.24030972,
            0.00216112,
            0.35723495,
            0.21323188,
            1.0467187,
            0.753764,
            0.37995362,
            -0.02775276,
            -0.33651975,
            1.7020398,
            0.35117954,
            0.9077985,
        ];
        for (actual, expected) in output.raw_actions.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        for (command, neutral) in output.commands.iter().zip(NEUTRAL_POSITIONS) {
            assert_eq!(command.position, neutral);
        }
        for _ in 1..WARMUP_CYCLES {
            let warming_output = policy.infer(&observation()).unwrap();
            for (command, neutral) in warming_output.commands.iter().zip(NEUTRAL_POSITIONS) {
                assert_eq!(command.position, neutral);
            }
        }
        // PyTorch model_40000.pt, eleven repeated observations, reference action processing.
        let expected_targets = [
            -0.13892317,
            0.62756664,
            -0.49325684,
            -0.28675380,
            0.03841142,
            0.77194327,
            0.07255333,
            -0.28302363,
            0.15023838,
            0.25162065,
            0.12582368,
            0.58391100,
            1.36589503,
            -0.47218060,
            0.79574788,
            -0.94130296,
            -0.26088044,
            -0.32409692,
            0.17914420,
            0.27197990,
        ];
        let active_output = policy.infer(&observation()).unwrap();
        for (command, expected) in active_output.commands.iter().zip(expected_targets) {
            let actual = command.position.get::<radian>();
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        policy.reset();
        assert_eq!(
            policy.state.history,
            [[0.0; POLICY_FRAME_SIZE]; POLICY_HISTORY_LENGTH]
        );
        assert_eq!(policy.state.previous_actions, [0.0; POLICY_JOINT_COUNT]);
        assert_eq!(policy.state.warmup_remaining, WARMUP_CYCLES);
        assert_eq!(policy.infer(&observation()).unwrap(), output);
    }
}
