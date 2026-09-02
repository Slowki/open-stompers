mod config;
mod gamepad;
mod policy;
mod visualization;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use rerun::external::anyhow::Context;
use rerun::{RecordingStream, RecordingStreamBuilder, Scalars};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;

use open_stompers_hardware::{
    GamepadCommand, InertialMeasurementSensor, Motor, PositionControlledMotor,
};
use open_stompers_microduck::{Microduck, MicroduckConfig};
use open_stompers_mini_pi::rsl_mlp_amp_inference::{
    NEUTRAL_POSITIONS, POLICY_JOINT_NAMES, POLICY_KD, POLICY_KP,
};
use open_stompers_mini_pi::transforms::pi_plus_s_12l8a0g2h1w::{
    JOINT_LIMITS as MINI_PI_PLUS_MODEL_JOINT_LIMITS, JOINT_NAMES as MINI_PI_PLUS_MODEL_JOINT_NAMES,
};
use open_stompers_mini_pi::{
    HEAD_POSITION_DERIVATIVE_GAIN, HEAD_POSITION_PROPORTIONAL_GAIN, JOINT_DEFINITIONS, MiniPiPlus,
    MiniPiPlusConfig, MiniPiPlusMotor, ReceivedMotorState, WAIST_POSITION_DERIVATIVE_GAIN,
    WAIST_POSITION_PROPORTIONAL_GAIN,
};

use open_stompers_mini_pi::velocity_policy::{
    JOINT_COUNT, ObservationContract, VelocityObservation,
};
#[cfg(test)]
use policy::MotorTarget;
use policy::{DeploymentOutput, DeploymentPolicy};

use config::{
    ConfigurationError, MiniPiPlusPolicyConfig, Platform, default_configuration_path,
    load_configuration,
};
use gamepad::GamepadInput;

const APPLICATION_ID: &str = "open-stompers";
const STARTUP_RAMP_DURATION: Duration = Duration::from_secs(2);
const STARTUP_NEUTRAL_HOLD_DURATION: Duration = Duration::from_secs(1);
const STARTUP_COMMAND_PERIOD: Duration = Duration::from_millis(20);

struct StartupMotor {
    joint_name: &'static str,
    motor: MiniPiPlusMotor,
    target: Option<(f64, f64, f64)>,
    initial_position: Option<f64>,
}

#[derive(Debug, Parser)]
#[command(version, about = "View Open Stompers robot telemetry")]
struct Arguments {
    /// Robot platform to use. Overrides the platform in the configuration.
    #[arg(long, value_enum)]
    platform: Option<Platform>,
    /// JSON configuration file. Defaults to the user configuration directory.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Save a Rerun recording to this file instead of serving it remotely.
    #[arg(long)]
    recording_path: Option<PathBuf>,
    /// Address to bind the Rerun gRPC server to.
    #[arg(long, default_value = "0.0.0.0")]
    bind_address: String,
    /// Port for the Rerun gRPC server.
    #[arg(long, default_value_t = 9876)]
    port: u16,
}

#[derive(Debug, Error)]
enum ApplicationError {
    #[error(transparent)]
    Configuration(#[from] ConfigurationError),
    #[error("Rerun initialization failed: {0}")]
    Rerun(#[from] rerun::RecordingStreamError),
    #[error("failed to initialize the Rerun recording file: {0}")]
    RecordingFile(String),
    #[error("Mini Pi Plus hardware failed: {0}")]
    Hardware(#[from] open_stompers_mini_pi::Error),
    #[error("Mini Pi Plus policy failed: {0}")]
    Policy(#[from] open_stompers_mini_pi::velocity_policy::VelocityPolicyError),
    #[error("Mini Pi Plus policy worker stopped")]
    PolicyWorkerStopped,
    #[error("no attached Mini Pi Plus motor exists for joint `{joint_name}`")]
    MissingPolicyMotor { joint_name: String },
    #[error("generated Mini Pi Plus model has no position limits for joint `{joint_name}`")]
    MissingMotorPositionLimits { joint_name: String },
    #[error("motor target for joint `{joint_name}` is not finite")]
    InvalidMotorTarget { joint_name: String },
    #[error("Microduck hardware failed: {0}")]
    Microduck(#[from] open_stompers_microduck::Error),
    #[error("interrupt handler failed: {0}")]
    Interrupt(#[from] std::io::Error),
    #[error("Rerun flush failed: {0}")]
    Flush(String),
}

fn initialize_recording(
    platform: Platform,
    recording_path: Option<&Path>,
    bind_address: &str,
    port: u16,
) -> Result<RecordingStream, ApplicationError> {
    let builder = RecordingStreamBuilder::new(APPLICATION_ID)
        .with_default_blueprint(visualization::default_blueprint(platform));
    Ok(if let Some(path) = recording_path {
        builder
            .save(path)
            .map_err(|error| ApplicationError::RecordingFile(error.to_string()))?
    } else {
        builder.serve_grpc_opts(bind_address, port, rerun::ServerOptions::default())?
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    rerun::external::re_log::setup_logging();
    let arguments = Arguments::parse();
    let (config_path, is_default) = match arguments.config {
        Some(path) => (path, false),
        None => (default_configuration_path()?, true),
    };
    let config = load_configuration(&config_path, is_default)?;
    let platform = arguments
        .platform
        .or(config.platform)
        .ok_or_else(|| ConfigurationError::PlatformRequired(config_path.clone()))?;

    let recording = initialize_recording(
        platform,
        arguments.recording_path.as_deref(),
        &arguments.bind_address,
        arguments.port,
    )?;

    match platform {
        Platform::MiniPiPlus => {
            run_mini_pi_plus(
                config.mini_pi_plus,
                config.mini_pi_plus_policy,
                &config_path,
                recording,
            )
            .await
        }
        Platform::Microduck => run_microduck(config.microduck, recording).await,
    }
}

async fn run_mini_pi_plus(
    configuration: MiniPiPlusConfig,
    policy_configuration: Option<MiniPiPlusPolicyConfig>,
    configuration_path: &Path,
    recording: RecordingStream,
) -> anyhow::Result<()> {
    let policy = policy_configuration
        .map(|policy_configuration| {
            let model_path = if policy_configuration.model_path.is_absolute() {
                policy_configuration.model_path
            } else {
                configuration_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(policy_configuration.model_path)
            };
            DeploymentPolicy::from_model_path(&model_path).with_context(|| {
                format!("Loading Mini Pi Plus policy model {}", model_path.display())
            })
        })
        .transpose()?;
    let mut robot = MiniPiPlus::open(configuration)
        .await
        .with_context(|| "Initializing robot hardware".to_string())?;
    let mut model = visualization::RobotModel::new(Platform::MiniPiPlus);
    if let Err(error) = model.log_meshes(&recording) {
        tracing::warn!("robot mesh visualization unavailable: {error}");
    }
    if policy.is_some() {
        log_policy_gains(&recording, policy.as_ref().map(DeploymentPolicy::contract))?;
    }
    let start = Instant::now();
    let runtime_result = async {
        if !move_mini_pi_plus_to_neutral(
            &mut robot,
            &recording,
            &mut model,
            start,
            policy.as_ref().map(DeploymentPolicy::contract),
        )
        .await?
        {
            return Ok(());
        }
        let gamepad = initialize_gamepad(policy.is_some());
        if policy.is_some() {
            robot.set_motor_telemetry_enabled(false).await?;
        }
        poll_mini_pi_plus(&mut robot, &recording, model, policy, gamepad, start).await
    }
    .await;
    let stop_result = robot.shutdown().await.map_err(ApplicationError::from);
    let flush_result = recording
        .flush_blocking()
        .map_err(|error| ApplicationError::Flush(error.to_string()));
    runtime_result?;
    stop_result?;
    flush_result?;
    Ok(())
}

async fn move_mini_pi_plus_to_neutral(
    robot: &mut MiniPiPlus,
    recording: &RecordingStream,
    model: &mut visualization::RobotModel,
    start: Instant,
    contract: Option<&ObservationContract>,
) -> Result<bool, ApplicationError> {
    let mut motors = robot
        .motor_names()
        .map(|joint_name| StartupMotor {
            joint_name,
            motor: robot
                .motor(joint_name)
                .expect("every attached Mini Pi Plus joint has a motor handle"),
            target: neutral_target(joint_name, contract),
            initial_position: None,
        })
        .collect::<Vec<_>>();
    let mut ticker = tokio::time::interval(STARTUP_COMMAND_PERIOD);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut reported_missing_telemetry = false;

    loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                tracing::info!("Startup homing interrupted before motor telemetry was available");
                return Ok(false);
            }
            _ = ticker.tick() => {
                let mut telemetry_ready = true;
                for startup_motor in &mut motors {
                    match startup_motor.motor.read_state().await {
                        Ok(state) => {
                            startup_motor.initial_position = Some(state.position.get::<radian>());
                        }
                        Err(open_stompers_mini_pi::Error::TelemetryUnavailable(_)) => {
                            telemetry_ready = false;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                if telemetry_ready {
                    break;
                }
                if !reported_missing_telemetry {
                    tracing::info!("Waiting for Mini Pi Plus motor telemetry before startup homing");
                    reported_missing_telemetry = true;
                }
            }
        }
    }

    let ramp_start = Instant::now();
    loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                tracing::info!("Startup homing interrupted");
                return Ok(false);
            }
            _ = ticker.tick() => {
                let ramp_elapsed = ramp_start.elapsed();
                let progress = startup_ramp_progress(ramp_elapsed);
                recording.set_duration_secs("elapsed", start.elapsed().as_secs_f64());
                for startup_motor in &mut motors {
                    let state = startup_motor.motor.read_state().await?;
                    let measured_position = state.position.get::<radian>();
                    log_scalar(
                        recording,
                        &format!("motors/{}/position_rad", startup_motor.joint_name),
                        measured_position,
                    )?;
                    model.record_joint(startup_motor.joint_name, measured_position);
                    if let Some((target_position, proportional_gain, derivative_gain)) =
                        startup_motor.target
                    {
                        let initial_position = startup_motor
                            .initial_position
                            .expect("startup motor telemetry was checked before ramping");
                        let target_position = interpolate_position(
                            initial_position,
                            target_position,
                            progress,
                        );
                        let target_position =
                            clamp_joint_position(startup_motor.joint_name, target_position)?;
                        startup_motor
                            .motor
                            .set_position_target(
                                uom::si::f64::Angle::new::<radian>(target_position),
                                proportional_gain,
                                derivative_gain,
                            )
                            .await?;
                    }
                }
                if let Err(error) = model.log_transforms(recording) {
                    tracing::warn!("robot model visualization failed during startup: {error}");
                }
                robot.flush_commands().await?;
                if startup_neutral_hold_complete(ramp_elapsed) {
                    tracing::info!("Mini Pi Plus startup homing held the configured neutral pose");
                    return Ok(true);
                }
            }
        }
    }
}

fn neutral_target(
    joint_name: &str,
    contract: Option<&ObservationContract>,
) -> Option<(f64, f64, f64)> {
    if let Some(contract) = contract {
        let index = contract
            .joint_names
            .iter()
            .position(|name| name == joint_name)?;
        return Some((
            f64::from(contract.home[index]),
            f64::from(contract.proportional_gains[index]),
            f64::from(contract.derivative_gains[index]),
        ));
    }
    amp_neutral_target(joint_name)
}

fn amp_neutral_target(joint_name: &str) -> Option<(f64, f64, f64)> {
    if let Some(index) = POLICY_JOINT_NAMES
        .iter()
        .position(|policy_joint_name| *policy_joint_name == joint_name)
    {
        return Some((
            NEUTRAL_POSITIONS[index].get::<radian>(),
            f64::from(POLICY_KP[index]),
            f64::from(POLICY_KD[index]),
        ));
    }

    match joint_name {
        "waist_yaw_joint" => Some((
            0.0,
            f64::from(WAIST_POSITION_PROPORTIONAL_GAIN),
            f64::from(WAIST_POSITION_DERIVATIVE_GAIN),
        )),
        "head_yaw_joint" | "head_pitch_joint" => Some((
            0.0,
            f64::from(HEAD_POSITION_PROPORTIONAL_GAIN),
            f64::from(HEAD_POSITION_DERIVATIVE_GAIN),
        )),
        _ => None,
    }
}

fn interpolate_position(initial_position: f64, target_position: f64, progress: f64) -> f64 {
    initial_position + (target_position - initial_position) * progress.clamp(0.0, 1.0)
}

fn startup_ramp_progress(elapsed: Duration) -> f64 {
    let linear_progress = (elapsed.as_secs_f64() / STARTUP_RAMP_DURATION.as_secs_f64()).min(1.0);
    linear_progress * linear_progress * (3.0 - 2.0 * linear_progress)
}

fn startup_neutral_hold_complete(elapsed: Duration) -> bool {
    elapsed >= STARTUP_RAMP_DURATION + STARTUP_NEUTRAL_HOLD_DURATION
}

fn initialize_gamepad(policy_enabled: bool) -> Option<GamepadInput> {
    if !policy_enabled {
        return None;
    }
    match GamepadInput::new() {
        Ok(gamepad) => Some(gamepad),
        Err(error) => {
            tracing::warn!("Gamepad input unavailable; policy commands will remain zero: {error}");
            None
        }
    }
}

async fn poll_mini_pi_plus(
    robot: &mut MiniPiPlus,
    recording: &RecordingStream,
    mut model: visualization::RobotModel,
    policy: Option<DeploymentPolicy>,
    mut gamepad: Option<GamepadInput>,
    start: Instant,
) -> Result<(), ApplicationError> {
    let policy_joint_names = policy
        .as_ref()
        .map(|policy| policy.joint_names().to_vec())
        .unwrap_or_default();
    let mut motors = robot
        .motor_names()
        .filter_map(|joint_name| robot.motor(joint_name).map(|motor| (joint_name, motor)))
        .collect::<Vec<_>>();
    let mut ticker = tokio::time::interval(std::time::Duration::from_millis(20));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let (log_sender, mut log_receiver) = mpsc::channel::<RuntimeLogRecord>(4);
    let dropped_log_records = Arc::new(AtomicU64::new(0));
    let logger_recording = recording.clone();
    let logger_dropped_records = Arc::clone(&dropped_log_records);
    let logger_task = tokio::task::spawn_blocking(move || {
        while let Some(record) = log_receiver.blocking_recv() {
            logger_recording.set_duration_secs("elapsed", record.elapsed_seconds);
            for (joint_name, position) in record.joint_positions {
                model.record_joint(joint_name, position);
            }
            if let Err(error) = model.log_transforms(&logger_recording) {
                tracing::warn!("robot model visualization failed: {error}");
            }
            for (path, value) in record.scalars {
                if let Err(error) = log_scalar(&logger_recording, &path, value) {
                    tracing::warn!("recording scalar {path} failed: {error}");
                }
            }
            let dropped = logger_dropped_records.swap(0, Ordering::Relaxed);
            if dropped > 0 {
                tracing::warn!(dropped, "diagnostic recording records were dropped");
            }
        }
        let dropped = logger_dropped_records.swap(0, Ordering::Relaxed);
        if dropped > 0 {
            tracing::warn!(dropped, "diagnostic recording records were dropped");
        }
    });
    let mut reported_missing_telemetry = false;
    let mut skipped_policy_cycles = 0_u64;
    let mut latency_overruns = 0_u64;
    let (policy_request_sender, mut policy_result_receiver, mut policy_pending) =
        if let Some(policy) = policy {
            let (request_sender, mut request_receiver) = mpsc::channel::<PolicyRequest>(1);
            let (result_sender, result_receiver) = mpsc::channel::<PolicyResult>(1);
            tokio::task::spawn_blocking(move || {
                let mut policy = policy;
                while let Some(request) = request_receiver.blocking_recv() {
                    match request {
                        PolicyRequest::Reset => policy.reset(),
                        PolicyRequest::Infer {
                            observation,
                            motor_data_received_at,
                            telemetry_acquisition_seconds,
                            oldest_input_age_seconds,
                            reply_spread_seconds,
                            submitted_at,
                        } => {
                            let inference_queue_delay_seconds =
                                submitted_at.elapsed().as_secs_f64();
                            let started = Instant::now();
                            let projected_gravity = observation.projected_gravity;
                            let result = policy.infer(&observation);
                            if result_sender
                                .blocking_send(PolicyResult {
                                    result,
                                    motor_data_received_at,
                                    inference_seconds: started.elapsed().as_secs_f64(),
                                    telemetry_acquisition_seconds,
                                    oldest_input_age_seconds,
                                    reply_spread_seconds,
                                    inference_queue_delay_seconds,
                                    result_handling_started_at: Instant::now(),
                                    projected_gravity,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
            });
            (Some(request_sender), Some(result_receiver), false)
        } else {
            (None, None, false)
        };

    loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                break;
            }
            _ = ticker.tick() => {
                let elapsed_seconds = start.elapsed().as_secs_f64();
                let policy_command = gamepad
                    .as_mut()
                    .map_or_else(GamepadCommand::default, GamepadInput::command);
                let mut measured_joints = std::collections::HashMap::with_capacity(motors.len());
                let mut log_record = RuntimeLogRecord {
                    elapsed_seconds,
                    scalars: Vec::with_capacity(motors.len() * 3 + 6),
                    joint_positions: Vec::with_capacity(motors.len()),
                };
                let mut policy_joint_states = std::collections::HashMap::with_capacity(policy_joint_names.len());
                let mut telemetry_acquisition_seconds = 0.0;
                if policy_request_sender.is_some() {
                    if policy_pending {
                        skipped_policy_cycles += 1;
                        continue;
                    }
                    let acquisition_started_at = Instant::now();
                    match robot.read_fresh_motor_states(&policy_joint_names, Duration::from_millis(10)).await {
                        Ok(states) => {
                            policy_joint_states.extend(policy_joint_names.iter().copied().zip(states));
                        }
                        Err(error) => {
                            tracing::warn!("Fresh policy telemetry unavailable: {error}");
                            if let Some(request_sender) = &policy_request_sender {
                                let _ = request_sender.try_send(PolicyRequest::Reset);
                            }
                            skipped_policy_cycles += 1;
                            continue;
                        }
                    }
                    telemetry_acquisition_seconds = acquisition_started_at.elapsed().as_secs_f64();
                }
                for (joint_name, motor) in &mut motors {
                    let state = match motor.read_state().await {
                        Ok(state) => state,
                        Err(open_stompers_mini_pi::Error::TelemetryUnavailable(_)) => {
                            if !reported_missing_telemetry {
                                tracing::warn!("No Mini Pi Plus motor telemetry is available yet");
                                reported_missing_telemetry = true;
                            }
                            continue
                        },
                        Err(error) => return Err(error.into()),
                    };
                    log_record.scalars.push((format!("motors/{joint_name}/position_rad"), state.position.get::<radian>()));
                    log_record.scalars.push((format!("motors/{joint_name}/velocity_rad_per_second"), state.velocity.get::<radian_per_second>()));
                    if let Some(torque) = state.torque {
                        log_record.scalars.push((format!("motors/{joint_name}/torque_newton_meters"), torque.get::<uom::si::torque::newton_meter>()));
                    }
                    log_record.joint_positions.push((joint_name, state.position.get::<radian>()));
                    measured_joints.insert(*joint_name, state);
                }
                let measurement = robot.read_measurement().await?;
                for axis_index in 0..3 {
                    let axis_name = ["x", "y", "z"][axis_index];
                    log_record.scalars.push((format!("imu/acceleration_{axis_name}_meters_per_second_squared"), measurement.acceleration[axis_index].get::<meter_per_second_squared>()));
                    log_record.scalars.push((format!("imu/angular_velocity_{axis_name}_radians_per_second"), measurement.angular_velocity[axis_index].get::<radian_per_second>()));
                }
                if log_sender.try_send(log_record).is_err() {
                    dropped_log_records.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(request_sender) = &policy_request_sender
                    && !policy_pending
                {
                    if let Some((observation, motor_data_received_at)) = build_policy_observation(robot, &policy_joint_states, policy_command, &policy_joint_names) {
                        let oldest_input_age_seconds = motor_data_received_at.elapsed().as_secs_f64();
                        let received_times = policy_joint_states.values().map(|state| state.received_at);
                        let earliest_reply = received_times.clone().min().unwrap_or(motor_data_received_at);
                        let latest_reply = received_times.max().unwrap_or(motor_data_received_at);
                        if request_sender.try_send(PolicyRequest::Infer {
                            observation: Box::new(observation),
                            motor_data_received_at,
                            telemetry_acquisition_seconds,
                            oldest_input_age_seconds,
                            reply_spread_seconds: latest_reply.duration_since(earliest_reply).as_secs_f64(),
                            submitted_at: Instant::now(),
                        }).is_ok() {
                            policy_pending = true;
                        }
                    } else {
                        let _ = request_sender.try_send(PolicyRequest::Reset);
                    }
                }
            }
            result = async { policy_result_receiver.as_mut().expect("policy receiver exists").recv().await }, if policy_pending => {
                policy_pending = false;
                let Some(result) = result else {
                    return Err(ApplicationError::PolicyWorkerStopped);
                };
                let motor_data_received_at = result.motor_data_received_at;
                let mut output = result.result?;
                clamp_policy_commands_to_motor_limits(&mut output)?;
                for command in &output.commands {
                    let (_, motor) = motors
                        .iter_mut()
                        .find(|(joint_name, _)| *joint_name == command.joint_name)
                        .ok_or_else(|| ApplicationError::MissingPolicyMotor {
                            joint_name: command.joint_name.to_owned(),
                        })?;
                    motor
                        .set_position_target(
                            command.position,
                            f64::from(command.proportional_gain),
                            f64::from(command.derivative_gain),
                        )
                        .await?;
                }
                let flush_started_at = Instant::now();
                robot.flush_commands().await?;
                let command_flush_seconds = flush_started_at.elapsed().as_secs_f64();
                let latency_seconds = motor_data_received_at.elapsed().as_secs_f64();
                let result_handling_seconds = result.result_handling_started_at.elapsed().as_secs_f64();
                if latency_seconds > 0.005 {
                    latency_overruns += 1;
                }
                let mut record = RuntimeLogRecord {
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                    scalars: vec![
                        ("policy/motor_data_to_command_latency_seconds".to_owned(), latency_seconds),
                        ("policy/telemetry_acquisition_seconds".to_owned(), result.telemetry_acquisition_seconds),
                        ("policy/oldest_input_age_seconds".to_owned(), result.oldest_input_age_seconds),
                        ("policy/reply_spread_seconds".to_owned(), result.reply_spread_seconds),
                        ("policy/inference_queue_delay_seconds".to_owned(), result.inference_queue_delay_seconds),
                        ("policy/result_handling_seconds".to_owned(), result_handling_seconds),
                        ("policy/command_flush_seconds".to_owned(), command_flush_seconds),
                        ("policy/latency_overrun_count".to_owned(), latency_overruns as f64),
                        ("policy/skipped_cycle_count".to_owned(), skipped_policy_cycles as f64),
                    ],
                    joint_positions: Vec::new(),
                };
                for (axis, value) in ["x", "y", "z"].into_iter().zip(result.projected_gravity) {
                    record.scalars.push((format!("policy/projected_gravity_{axis}"), f64::from(value)));
                }
                collect_policy_output(&mut record.scalars, &output, result.inference_seconds);
                if log_sender.try_send(record).is_err() {
                    dropped_log_records.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
    drop(log_sender);
    if let Err(error) = logger_task.await {
        tracing::warn!("diagnostic recording worker failed: {error}");
    }
    Ok(())
}

struct RuntimeLogRecord {
    elapsed_seconds: f64,
    scalars: Vec<(String, f64)>,
    joint_positions: Vec<(&'static str, f64)>,
}

enum PolicyRequest {
    Reset,
    Infer {
        observation: Box<VelocityObservation>,
        motor_data_received_at: Instant,
        telemetry_acquisition_seconds: f64,
        oldest_input_age_seconds: f64,
        reply_spread_seconds: f64,
        submitted_at: Instant,
    },
}

struct PolicyResult {
    result: Result<DeploymentOutput, open_stompers_mini_pi::velocity_policy::VelocityPolicyError>,
    motor_data_received_at: Instant,
    inference_seconds: f64,
    telemetry_acquisition_seconds: f64,
    oldest_input_age_seconds: f64,
    reply_spread_seconds: f64,
    inference_queue_delay_seconds: f64,
    result_handling_started_at: Instant,
    projected_gravity: [f32; 3],
}

#[allow(clippy::cast_possible_truncation)]
fn build_policy_observation(
    robot: &MiniPiPlus,
    measured_joints: &std::collections::HashMap<&'static str, ReceivedMotorState>,
    command: GamepadCommand,
    joint_names: &[&'static str],
) -> Option<(VelocityObservation, Instant)> {
    let snapshot = robot.read_inertial_snapshot().ok()?;
    if snapshot.angular_velocity_received_at?.elapsed() > Duration::from_millis(100)
        || snapshot.orientation_received_at?.elapsed() > Duration::from_millis(100)
    {
        return None;
    }
    let mut joint_positions = [0.0; JOINT_COUNT];
    let mut joint_velocities = [0.0; JOINT_COUNT];
    let mut motor_data_received_at = Instant::now();
    for (index, joint_name) in joint_names.iter().enumerate() {
        let snapshot = measured_joints.get(joint_name)?;
        let state = snapshot.state;
        let received_at = snapshot.received_at;
        if received_at.elapsed() > Duration::from_millis(100) {
            return None;
        }
        motor_data_received_at = motor_data_received_at.min(received_at);
        joint_positions[index] = state.position.get::<radian>();
        joint_velocities[index] = state.velocity.get::<radian_per_second>();
    }
    let angular_velocity = snapshot
        .measurement
        .angular_velocity
        .map(|value| value.get::<radian_per_second>() as f32);
    let projected_gravity = projected_gravity_from_quaternion(snapshot.orientation_wxyz?);
    Some((
        VelocityObservation {
            angular_velocity,
            projected_gravity,
            desired_velocity: scale_velocity_command(command),
            joint_positions,
            joint_velocities,
        },
        motor_data_received_at,
    ))
}

fn scale_velocity_command(command: GamepadCommand) -> [f32; 3] {
    let finite_axis = |value: f32| {
        if value.is_finite() {
            value.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    };
    let forward = finite_axis(command.forward_axis);
    [
        forward * if forward >= 0.0 { 1.5 } else { 0.3 },
        finite_axis(command.lateral_axis) * 0.2,
        finite_axis(command.yaw_axis) * 0.8,
    ]
}

fn projected_gravity_from_quaternion(quaternion: [f64; 4]) -> [f32; 3] {
    let [w, x, y, z] = quaternion;
    [
        (-2.0 * x.mul_add(z, -(w * y))) as f32,
        (-2.0 * y.mul_add(z, w * x)) as f32,
        2.0f64.mul_add(x.mul_add(x, y * y), -1.0) as f32,
    ]
}

fn clamp_joint_position(joint_name: &str, position_rad: f64) -> Result<f64, ApplicationError> {
    let Some(joint_index) = MINI_PI_PLUS_MODEL_JOINT_NAMES
        .iter()
        .position(|model_joint_name| *model_joint_name == joint_name)
    else {
        return Err(ApplicationError::MissingMotorPositionLimits {
            joint_name: joint_name.to_owned(),
        });
    };
    let Some(limits) = MINI_PI_PLUS_MODEL_JOINT_LIMITS[joint_index] else {
        return Err(ApplicationError::MissingMotorPositionLimits {
            joint_name: joint_name.to_owned(),
        });
    };
    let lower_limit = limits.minimum_position.get::<radian>();
    let upper_limit = limits.maximum_position.get::<radian>();
    if !position_rad.is_finite() {
        return Err(ApplicationError::InvalidMotorTarget {
            joint_name: joint_name.to_owned(),
        });
    }
    let clamped_position_rad = position_rad.clamp(lower_limit, upper_limit);
    if clamped_position_rad != position_rad {
        tracing::warn!(
            joint_name,
            requested_position_rad = position_rad,
            clamped_position_rad,
            lower_limit_rad = lower_limit,
            upper_limit_rad = upper_limit,
            "Clamping Mini Pi Plus motor target to its configured position range"
        );
    }
    Ok(clamped_position_rad)
}

#[allow(clippy::cast_possible_truncation)]
fn clamp_policy_commands_to_motor_limits(
    output: &mut DeploymentOutput,
) -> Result<(), ApplicationError> {
    for command in &mut output.commands {
        let clamped_position =
            clamp_joint_position(command.joint_name, command.position.get::<radian>())?;
        command.position = uom::si::f64::Angle::new::<radian>(clamped_position);
    }
    Ok(())
}

fn collect_policy_output(
    scalars: &mut Vec<(String, f64)>,
    output: &DeploymentOutput,
    inference_seconds: f64,
) {
    scalars.push((
        "policy/inference_duration_seconds".to_owned(),
        inference_seconds,
    ));
    for (command, action) in output.commands.iter().zip(output.policy_actions) {
        scalars.push((
            format!("policy/actions/{}/sampled", command.joint_name),
            f64::from(action),
        ));
    }
    for command in output.commands {
        scalars.push((
            format!("policy/commands/{}/position_rad", command.joint_name),
            command.position.get::<radian>(),
        ));
    }
}

#[cfg(test)]
fn log_policy_output(
    recording: &RecordingStream,
    output: &DeploymentOutput,
    inference_seconds: f64,
) -> Result<(), ApplicationError> {
    let mut scalars = Vec::new();
    collect_policy_output(&mut scalars, output, inference_seconds);
    for (path, value) in scalars {
        log_scalar(recording, &path, value)?;
    }
    Ok(())
}

fn log_policy_gains(
    recording: &RecordingStream,
    contract: Option<&ObservationContract>,
) -> Result<(), ApplicationError> {
    if let Some(contract) = contract {
        for (index, joint) in contract.joint_names.iter().enumerate() {
            log_scalar(
                recording,
                &format!("policy/config/{joint}/kp"),
                f64::from(contract.proportional_gains[index]),
            )?;
            log_scalar(
                recording,
                &format!("policy/config/{joint}/kd"),
                f64::from(contract.derivative_gains[index]),
            )?;
        }
        return Ok(());
    }
    for joint in JOINT_DEFINITIONS.iter().filter(|joint| joint.attached) {
        log_scalar(
            recording,
            &format!("policy/config/{}/kp", joint.name),
            f64::from(joint.proportional_gain),
        )?;
        log_scalar(
            recording,
            &format!("policy/config/{}/kd", joint.name),
            f64::from(joint.derivative_gain),
        )?;
    }
    Ok(())
}

async fn run_microduck(
    configuration: MicroduckConfig,
    recording: RecordingStream,
) -> anyhow::Result<()> {
    let mut robot = Microduck::open(configuration)
        .await
        .with_context(|| "Initializing robot hardware".to_string())?;
    let mut model = visualization::RobotModel::new(Platform::Microduck);
    if let Err(error) = model.log_meshes(&recording) {
        tracing::warn!("robot mesh visualization unavailable: {error}");
    }
    let runtime_result = poll_microduck(&mut robot, &recording, &mut model).await;
    let stop_result = robot.shutdown().await.map_err(ApplicationError::from);
    let flush_result = recording
        .flush_blocking()
        .map_err(|error| ApplicationError::Flush(error.to_string()));
    runtime_result?;
    stop_result?;
    flush_result?;
    Ok(())
}

async fn poll_microduck(
    robot: &mut Microduck,
    recording: &RecordingStream,
    model: &mut visualization::RobotModel,
) -> Result<(), ApplicationError> {
    let mut motors = robot
        .motor_names()
        .filter_map(|joint_name| robot.motor(joint_name).map(|motor| (joint_name, motor)))
        .collect::<Vec<_>>();
    let start = Instant::now();
    let mut ticker = tokio::time::interval(std::time::Duration::from_millis(20));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                break;
            }
            _ = ticker.tick() => {
                let elapsed_seconds = start.elapsed().as_secs_f64();
                recording.set_duration_secs("elapsed", elapsed_seconds);
                for (joint_name, motor) in &mut motors {
                    let state = match motor.read_state().await {
                        Ok(state) => state,
                        Err(open_stompers_microduck::Error::TelemetryUnavailable(_)) => {
                            tracing::warn!("No telemetry available yet");
                            continue
                        },
                        Err(error) => return Err(error.into()),
                    };
                    log_scalar(recording, &format!("motors/{joint_name}/position_rad"), state.position.get::<radian>())?;
                    log_scalar(recording, &format!("motors/{joint_name}/velocity_rad_per_second"), state.velocity.get::<radian_per_second>())?;
                    if let Some(torque) = state.torque {
                        log_scalar(recording, &format!("motors/{joint_name}/torque_newton_meters"), torque.get::<uom::si::torque::newton_meter>())?;
                    }
                    model.record_joint(joint_name, state.position.get::<radian>());
                }
                if let Err(error) = model.log_transforms(recording) {
                    tracing::warn!("robot model visualization failed: {error}");
                }
                let measurement = robot.read_measurement().await?;
                for axis_index in 0..3 {
                    let axis_name = ["x", "y", "z"][axis_index];
                    log_scalar(recording, &format!("imu/acceleration_{axis_name}_meters_per_second_squared"), measurement.acceleration[axis_index].get::<meter_per_second_squared>())?;
                    log_scalar(recording, &format!("imu/angular_velocity_{axis_name}_radians_per_second"), measurement.angular_velocity[axis_index].get::<radian_per_second>())?;
                }
                if let Some(orientation) = robot.latest_orientation_wxyz() {
                    for (name, value) in ["w", "x", "y", "z"].into_iter().zip(orientation) {
                        log_scalar(recording, &format!("imu/orientation_{name}"), value)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn log_scalar(
    recording: &RecordingStream,
    entity_path: &str,
    value: f64,
) -> Result<(), ApplicationError> {
    recording.log(entity_path, &Scalars::single(value))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_commands_are_written_to_a_rerun_recording() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("policy.rrd");
        let recording = RecordingStreamBuilder::new(APPLICATION_ID)
            .save(&path)
            .unwrap();
        let commands = std::array::from_fn(|index| MotorTarget {
            joint_name: MINI_PI_PLUS_MODEL_JOINT_NAMES[index],
            position: uom::si::f64::Angle::new::<radian>(0.1),
            proportional_gain: 1.0,
            derivative_gain: 0.1,
        });
        let output = DeploymentOutput {
            policy_actions: [0.0; JOINT_COUNT],
            commands,
        };

        log_policy_gains(&recording, None).unwrap();
        log_policy_output(&recording, &output, 0.001).unwrap();
        recording.flush_blocking().unwrap();

        assert!(std::fs::metadata(path).unwrap().len() > 0);
    }

    #[test]
    fn policy_motor_targets_are_clamped_to_generated_joint_limits() {
        let commands = std::array::from_fn(|index| MotorTarget {
            joint_name: MINI_PI_PLUS_MODEL_JOINT_NAMES[index],
            position: uom::si::f64::Angle::new::<radian>(0.0),
            proportional_gain: 1.0,
            derivative_gain: 0.1,
        });
        let mut output = DeploymentOutput {
            policy_actions: [0.0; JOINT_COUNT],
            commands,
        };
        let cases = [
            ("head_yaw_joint", 2.0, 1.43),
            ("l_hip_roll_joint", -0.25, -0.17),
            ("r_hip_roll_joint", 0.25, 0.17),
            ("l_ankle_pitch_joint", 3.15, 0.92),
        ];
        for (name, requested, _) in cases {
            let target = output
                .commands
                .iter_mut()
                .find(|target| target.joint_name == name)
                .unwrap();
            target.position = uom::si::f64::Angle::new::<radian>(requested);
        }
        clamp_policy_commands_to_motor_limits(&mut output).unwrap();
        for (name, _, expected) in cases {
            let target = output
                .commands
                .iter()
                .find(|target| target.joint_name == name)
                .unwrap();
            assert_eq!(target.position.get::<radian>(), expected);
        }
        for joint in JOINT_DEFINITIONS.iter().filter(|joint| joint.attached) {
            assert!(
                MINI_PI_PLUS_MODEL_JOINT_NAMES
                    .iter()
                    .position(|model_joint_name| *model_joint_name == joint.name)
                    .is_some_and(|index| MINI_PI_PLUS_MODEL_JOINT_LIMITS[index].is_some())
            );
        }
    }

    #[test]
    fn startup_targets_cover_attached_joints_with_amp_neutral_targets_and_gains() {
        let attached_joints = JOINT_DEFINITIONS
            .iter()
            .filter(|joint| joint.attached)
            .collect::<Vec<_>>();
        assert_eq!(attached_joints.len(), 23);
        assert_eq!(
            attached_joints
                .iter()
                .filter(|joint| amp_neutral_target(joint.name).is_some())
                .count(),
            POLICY_JOINT_NAMES.len() + 3
        );

        for joint in attached_joints {
            if let Some(index) = POLICY_JOINT_NAMES
                .iter()
                .position(|policy_joint_name| *policy_joint_name == joint.name)
            {
                let (target_position, proportional_gain, derivative_gain) =
                    amp_neutral_target(joint.name).expect("actor joint should have a target");
                assert_eq!(target_position, NEUTRAL_POSITIONS[index].get::<radian>());
                assert_eq!(proportional_gain, f64::from(POLICY_KP[index]));
                assert_eq!(derivative_gain, f64::from(POLICY_KD[index]));
            } else {
                match joint.name {
                    "waist_yaw_joint" => {
                        assert_eq!(
                            amp_neutral_target(joint.name),
                            Some((
                                0.0,
                                f64::from(WAIST_POSITION_PROPORTIONAL_GAIN),
                                f64::from(WAIST_POSITION_DERIVATIVE_GAIN),
                            ))
                        );
                    }
                    "head_yaw_joint" | "head_pitch_joint" => {
                        assert_eq!(
                            amp_neutral_target(joint.name),
                            Some((
                                0.0,
                                f64::from(HEAD_POSITION_PROPORTIONAL_GAIN),
                                f64::from(HEAD_POSITION_DERIVATIVE_GAIN),
                            ))
                        );
                    }
                    _ => panic!("unexpected attached joint outside the AMP actor"),
                }
            }
        }
    }

    #[test]
    fn startup_position_interpolation_reaches_targets_over_the_ramp() {
        assert_eq!(interpolate_position(1.0, -1.0, 0.0), 1.0);
        assert_eq!(interpolate_position(1.0, -1.0, 0.5), 0.0);
        assert_eq!(interpolate_position(1.0, -1.0, 1.0), -1.0);
        assert_eq!(interpolate_position(-0.25, -0.25, 0.5), -0.25);
        assert_eq!(interpolate_position(1.0, -1.0, 2.0), -1.0);
    }

    #[test]
    fn startup_holds_neutral_for_two_seconds_after_the_ramp() {
        assert_eq!(startup_ramp_progress(Duration::ZERO), 0.0);
        assert_eq!(startup_ramp_progress(STARTUP_RAMP_DURATION / 2), 0.5);
        assert_eq!(startup_ramp_progress(STARTUP_RAMP_DURATION), 1.0);
        assert!(!startup_neutral_hold_complete(STARTUP_RAMP_DURATION));
        assert!(!startup_neutral_hold_complete(
            STARTUP_RAMP_DURATION + STARTUP_NEUTRAL_HOLD_DURATION - Duration::from_millis(1)
        ));
        assert!(startup_neutral_hold_complete(
            STARTUP_RAMP_DURATION + STARTUP_NEUTRAL_HOLD_DURATION
        ));
    }
}
