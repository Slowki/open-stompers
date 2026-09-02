//! Mini Pi Plus motor and inertial measurement hardware.
//!
//! Opening a robot starts background reception and periodic motor requests.
//! Clone robot and motor handles into Tokio tasks; hardware trait reads return
//! cached values immediately. Motor telemetry is unavailable before the first
//! reply, and receive timestamps let callers assess subsequent freshness.
//! Await [`MiniPiPlus::shutdown`] to stop motors and join communication tasks.
//!
//! ```no_run
//! use open_stompers_hardware::Motor;
//! use open_stompers_mini_pi::{Error, MiniPiPlus, MiniPiPlusConfig};
//!
//! # async fn run() -> open_stompers_mini_pi::Result<()> {
//! let mut robot = MiniPiPlus::open(MiniPiPlusConfig::default()).await?;
//! let joint_name = robot.motor_names().next().expect("robot has motors");
//! let mut motor = robot.motor(joint_name).expect("attached motor");
//! let state = tokio::spawn(async move {
//!     match motor.read_state().await {
//!         Ok(state) => Ok(Some(state)),
//!         Err(Error::TelemetryUnavailable(_)) => Ok(None),
//!         Err(error) => Err(error),
//!     }
//! }).await?;
//! robot.shutdown().await?;
//! state?;
//! # Ok(())
//! # }
//! ```

use std::array;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crate::config::{MiniPiPlusConfig, SERIAL_PORT_COUNT};
use open_stompers_hardware::{
    InertialMeasurement, InertialMeasurementSensor, InertialSnapshot, Motor, MotorState,
    PositionControlledMotor,
};
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f32::{
    Angle as JointAngle, AngularVelocity as JointAngularVelocity, Torque as JointTorque,
};
use uom::si::f64::{Angle, AngularVelocity, Torque};
use uom::si::torque::newton_meter;

use crate::background::{BackgroundDriver, ReceivedMotorState};
use crate::controller::{FunVersion, MotorController};
use crate::error::{Error, Result};
use crate::motor::{MotorConfig, MotorType};
use crate::yesense_driver::YesenseDriver;

const MOTOR_COMMAND_TIMEOUT: Duration = Duration::from_millis(50);

/// Head position proportional gain.
pub const HEAD_POSITION_PROPORTIONAL_GAIN: f32 = 0.5;
/// Head position derivative gain.
pub const HEAD_POSITION_DERIVATIVE_GAIN: f32 = 0.005;
/// Waist position proportional gain.
pub const WAIST_POSITION_PROPORTIONAL_GAIN: f32 = 1.0;
/// Waist position derivative gain.
pub const WAIST_POSITION_DERIVATIVE_GAIN: f32 = 0.1;

/// Static Mini Pi Plus joint-to-motor wiring description.
#[derive(Clone, Copy, Debug)]
pub struct JointDefinition {
    /// URDF joint name.
    pub name: &'static str,
    /// Zero-based motor serial port index.
    pub port_index: usize,
    /// Motor configuration including motor id, type, reversal, and offset.
    pub motor: MotorConfig,
    /// Whether a physical motor is attached to this motor slot.
    pub attached: bool,
    /// Production proportional gain.
    pub proportional_gain: f32,
    /// Production derivative gain.
    pub derivative_gain: f32,
}

impl JointDefinition {
    const fn new(
        name: &'static str,
        port_index: usize,
        id: u8,
        motor_type: MotorType,
        reverse: bool,
        offset_rad: f32,
        attached: bool,
    ) -> Self {
        Self {
            name,
            port_index,
            motor: MotorConfig {
                id,
                motor_type,
                reverse,
                offset_rad,
            },
            attached,
            proportional_gain: motor_type.position_proportional_gain(),
            derivative_gain: motor_type.velocity_derivative_gain(),
        }
    }

    const fn with_position_gains(mut self, proportional_gain: f32, derivative_gain: f32) -> Self {
        self.proportional_gain = proportional_gain;
        self.derivative_gain = derivative_gain;
        self
    }
}

/// Motor topology matching the Mini Pi Plus OEM controller configuration.
pub const JOINT_DEFINITIONS: [JointDefinition; 25] = [
    JointDefinition::new(
        "l_ankle_roll_joint",
        0,
        1,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    ),
    JointDefinition::new(
        "l_ankle_pitch_joint",
        0,
        2,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    )
    .with_position_gains(210.0, 1.1),
    JointDefinition::new("l_calf_joint", 0, 3, MotorType::M5047_36_2, true, 0.0, true),
    JointDefinition::new(
        "l_thigh_joint",
        0,
        4,
        MotorType::M5047_36_2,
        true,
        0.0,
        true,
    ),
    JointDefinition::new(
        "l_hip_roll_joint",
        0,
        5,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    )
    .with_position_gains(110.0, 1.4),
    JointDefinition::new(
        "l_hip_pitch_joint",
        0,
        6,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    ),
    JointDefinition::new(
        "r_ankle_roll_joint",
        1,
        1,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    ),
    JointDefinition::new(
        "r_ankle_pitch_joint",
        1,
        2,
        MotorType::M5047_36_2,
        true,
        0.0,
        true,
    )
    .with_position_gains(210.0, 1.1),
    JointDefinition::new(
        "r_calf_joint",
        1,
        3,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    ),
    JointDefinition::new(
        "r_thigh_joint",
        1,
        4,
        MotorType::M5047_36_2,
        true,
        0.0,
        true,
    ),
    JointDefinition::new(
        "r_hip_roll_joint",
        1,
        5,
        MotorType::M5047_36_2,
        false,
        0.0,
        true,
    )
    .with_position_gains(110.0, 1.4),
    JointDefinition::new(
        "r_hip_pitch_joint",
        1,
        6,
        MotorType::M5047_36_2,
        true,
        0.0,
        true,
    ),
    JointDefinition::new(
        "l_upper_arm_joint",
        2,
        3,
        MotorType::M4438_30,
        true,
        0.0,
        true,
    ),
    JointDefinition::new(
        "l_shoulder_roll_joint",
        2,
        2,
        MotorType::M4438_30,
        true,
        -std::f32::consts::FRAC_PI_2,
        true,
    ),
    JointDefinition::new(
        "l_shoulder_pitch_joint",
        2,
        1,
        MotorType::M4438_30,
        false,
        std::f32::consts::FRAC_PI_2,
        true,
    ),
    JointDefinition::new("l_elbow_joint", 2, 4, MotorType::M4438_30, true, 0.0, true),
    JointDefinition::new(
        "l_wrist_joint",
        2,
        5,
        MotorType::M4438_30,
        false,
        0.0,
        false,
    ),
    JointDefinition::new(
        "r_shoulder_pitch_joint",
        3,
        1,
        MotorType::M4438_30,
        true,
        std::f32::consts::FRAC_PI_2,
        true,
    ),
    JointDefinition::new(
        "r_shoulder_roll_joint",
        3,
        2,
        MotorType::M4438_30,
        true,
        std::f32::consts::FRAC_PI_2,
        true,
    ),
    JointDefinition::new(
        "r_upper_arm_joint",
        3,
        3,
        MotorType::M4438_30,
        true,
        0.0,
        true,
    ),
    JointDefinition::new("r_elbow_joint", 3, 4, MotorType::M4438_30, false, 0.0, true),
    JointDefinition::new(
        "r_wrist_joint",
        3,
        5,
        MotorType::M4438_30,
        false,
        0.0,
        false,
    ),
    JointDefinition::new(
        "head_yaw_joint",
        4,
        1,
        MotorType::M3536_32,
        false,
        0.0,
        true,
    )
    .with_position_gains(
        HEAD_POSITION_PROPORTIONAL_GAIN,
        HEAD_POSITION_DERIVATIVE_GAIN,
    ),
    JointDefinition::new(
        "head_pitch_joint",
        4,
        2,
        MotorType::M3536_32,
        false,
        0.0,
        true,
    )
    .with_position_gains(
        HEAD_POSITION_PROPORTIONAL_GAIN,
        HEAD_POSITION_DERIVATIVE_GAIN,
    ),
    JointDefinition::new(
        "waist_yaw_joint",
        5,
        1,
        MotorType::M3536_32,
        false,
        0.0,
        true,
    )
    .with_position_gains(
        WAIST_POSITION_PROPORTIONAL_GAIN,
        WAIST_POSITION_DERIVATIVE_GAIN,
    ),
];

/// Mini Pi Plus robot coordinator for batched motor and IMU I/O.
#[derive(Clone)]
pub struct MiniPiPlus {
    driver: Arc<BackgroundDriver>,
}

struct Driver {
    controllers: Vec<MotorController>,
}

/// Handle for one attached Mini Pi Plus joint motor.
#[derive(Clone)]
pub struct MiniPiPlusMotor {
    driver: Arc<BackgroundDriver>,
    joint_name: &'static str,
    port_index: usize,
    motor: MotorConfig,
}

impl MiniPiPlus {
    /// Initialize motor boards and the sensor, then start background communication.
    pub async fn open(configuration: MiniPiPlusConfig) -> Result<Self> {
        if configuration.motor_telemetry_period == std::time::Duration::from_millis(0) {
            return Err(Error::InvalidTelemetryPeriod);
        }
        let mut motors_per_port: [Vec<MotorConfig>; SERIAL_PORT_COUNT] =
            array::from_fn(|_| Vec::new());
        for joint in JOINT_DEFINITIONS {
            motors_per_port[joint.port_index].push(joint.motor);
        }

        let [motors_0, motors_1, motors_2, motors_3, motors_4, motors_5] = motors_per_port;
        let controller_0 = MotorController::open(
            &configuration.motor_serial_paths[0],
            configuration.motor_baudrate,
            motors_0,
        )?;
        let controller_1 = MotorController::open(
            &configuration.motor_serial_paths[1],
            configuration.motor_baudrate,
            motors_1,
        )?;
        let controller_2 = MotorController::open(
            &configuration.motor_serial_paths[2],
            configuration.motor_baudrate,
            motors_2,
        )?;
        let controller_3 = MotorController::open(
            &configuration.motor_serial_paths[3],
            configuration.motor_baudrate,
            motors_3,
        )?;
        let controller_4 = MotorController::open(
            &configuration.motor_serial_paths[4],
            configuration.motor_baudrate,
            motors_4,
        )?;
        let controller_5 = MotorController::open(
            &configuration.motor_serial_paths[5],
            configuration.motor_baudrate,
            motors_5,
        )?;
        let imu = YesenseDriver::open(&configuration.imu_serial_path, configuration.imu_baudrate)?;

        let mut driver = Driver {
            controllers: vec![
                controller_0,
                controller_1,
                controller_2,
                controller_3,
                controller_4,
                controller_5,
            ],
        };
        driver.initialize().await?;
        Ok(Self {
            driver: Arc::new(BackgroundDriver::start(
                driver.controllers,
                imu,
                configuration.motor_telemetry_period,
            )),
        })
    }

    /// Return a handle for an attached joint motor by URDF name.
    #[must_use]
    pub fn motor(&self, joint_name: &str) -> Option<MiniPiPlusMotor> {
        let definition = JOINT_DEFINITIONS
            .iter()
            .find(|joint| joint.attached && joint.name == joint_name)?;
        Some(MiniPiPlusMotor {
            driver: Arc::clone(&self.driver),
            joint_name: definition.name,
            port_index: definition.port_index,
            motor: definition.motor,
        })
    }

    /// List the names of attached joints exposed as motors.
    pub fn motor_names(&self) -> impl Iterator<Item = &'static str> {
        JOINT_DEFINITIONS
            .iter()
            .filter(|joint| joint.attached)
            .map(|joint| joint.name)
    }

    /// Flush all staged motor commands, batching writes by serial board.
    pub async fn flush_commands(&mut self) -> Result<()> {
        self.driver.flush_all().await
    }

    /// Return the latest inertial sample and policy telemetry receive times.
    pub fn read_inertial_snapshot(&self) -> Result<InertialSnapshot> {
        self.driver.read_inertial_snapshot()
    }

    /// Request motor telemetry and wait for fresh states from every named joint.
    pub async fn read_fresh_motor_states(
        &self,
        joint_names: &[&str],
        timeout: Duration,
    ) -> Result<Vec<ReceivedMotorState>> {
        let mut motors = Vec::with_capacity(joint_names.len());
        for joint_name in joint_names {
            let definition = JOINT_DEFINITIONS
                .iter()
                .find(|definition| definition.attached && definition.name == *joint_name)
                .ok_or_else(|| Error::UnknownJoint((*joint_name).to_owned()))?;
            motors.push((definition.port_index, definition.motor.id));
        }
        self.driver.read_fresh_motor_states(&motors, timeout).await
    }

    /// Enable or suspend periodic motor requests.
    pub async fn set_motor_telemetry_enabled(&self, enabled: bool) -> Result<()> {
        self.driver.set_motor_telemetry_enabled(enabled).await
    }

    /// Stop motors on every communication board.
    pub async fn stop_all(&mut self) -> Result<()> {
        self.driver.stop_all().await
    }

    /// Stop motors, close communication tasks, and wait for them to finish.
    ///
    /// Shut down every clone and motor handle sharing this connection.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.driver.shutdown().await
    }
}

impl MiniPiPlusMotor {
    /// Return the receive time of the latest valid reply for this motor.
    ///
    /// Return `None` before the first reply. Preserve the timestamp during
    /// reply gaps and after shutdown; check `read_state` for connection errors.
    #[must_use]
    pub fn latest_state_received_at(&self) -> Option<Instant> {
        self.driver.boards[self.port_index]
            .snapshot(self.motor.id)
            .map(|snapshot| snapshot.received_at)
    }
}

impl InertialMeasurementSensor for MiniPiPlus {
    type Error = Error;

    async fn read_measurement(&mut self) -> Result<InertialMeasurement> {
        self.driver.read_measurement()
    }
}

impl Motor for MiniPiPlusMotor {
    type Error = Error;

    async fn read_state(&mut self) -> Result<MotorState> {
        self.driver.check_open()?;
        let board = &self.driver.boards[self.port_index];
        board.check_health()?;
        let state = board
            .snapshot(self.motor.id)
            .ok_or_else(|| Error::TelemetryUnavailable(self.joint_name.to_owned()))?
            .state;
        Ok(MotorState {
            position: Angle::new::<radian>(f64::from(state.position.get::<radian>())),
            velocity: AngularVelocity::new::<radian_per_second>(f64::from(
                state.velocity.get::<radian_per_second>(),
            )),
            torque: Some(Torque::new::<newton_meter>(f64::from(
                state.torque.get::<newton_meter>(),
            ))),
        })
    }
}

impl PositionControlledMotor for MiniPiPlusMotor {
    async fn set_position_target(&mut self, target: Angle, kp: f64, kd: f64) -> Result<()> {
        let position = target.get::<radian>();
        if !position.is_finite() || !kp.is_finite() || !kd.is_finite() || kp < 0.0 || kd < 0.0 {
            return Err(Error::InvalidCommandValue(self.joint_name.to_owned()));
        }
        if position.abs() > f64::from(f32::MAX)
            || kp > f64::from(f32::MAX)
            || kd > f64::from(f32::MAX)
        {
            return Err(Error::InvalidCommandValue(self.joint_name.to_owned()));
        }

        self.driver.check_open()?;
        let board = &self.driver.boards[self.port_index];
        board.check_health()?;
        board
            .pending
            .lock()
            .expect("motor command lock poisoned")
            .stage_pos_vel_torque_kp_kd(
                self.motor,
                JointAngle::new::<radian>(position as f32),
                JointAngularVelocity::new::<radian_per_second>(0.0),
                JointTorque::new::<newton_meter>(0.0),
                kp as f32,
                kd as f32,
            );
        Ok(())
    }
}

impl Driver {
    async fn initialize(&mut self) -> Result<()> {
        self.set_motor_counts().await?;
        self.fdcan_reset_all().await?;
        self.set_fun_version_all(FunVersion::V5).await?;
        self.set_all_timeouts(MOTOR_COMMAND_TIMEOUT).await?;
        self.stop_all().await
    }

    async fn set_motor_counts(&mut self) -> Result<()> {
        let mut counts = [0_u8; SERIAL_PORT_COUNT];
        for joint in JOINT_DEFINITIONS {
            counts[joint.port_index] += 1;
        }
        let [
            controller_0,
            controller_1,
            controller_2,
            controller_3,
            controller_4,
            controller_5,
        ] = controllers_as_array(&mut self.controllers);
        tokio::try_join!(
            controller_0.set_motor_count(counts[0]),
            controller_1.set_motor_count(counts[1]),
            controller_2.set_motor_count(counts[2]),
            controller_3.set_motor_count(counts[3]),
            controller_4.set_motor_count(counts[4]),
            controller_5.set_motor_count(counts[5]),
        )?;
        Ok(())
    }

    async fn fdcan_reset_all(&mut self) -> Result<()> {
        let [
            controller_0,
            controller_1,
            controller_2,
            controller_3,
            controller_4,
            controller_5,
        ] = controllers_as_array(&mut self.controllers);
        tokio::try_join!(
            controller_0.fdcan_reset(),
            controller_1.fdcan_reset(),
            controller_2.fdcan_reset(),
            controller_3.fdcan_reset(),
            controller_4.fdcan_reset(),
            controller_5.fdcan_reset(),
        )?;
        Ok(())
    }

    async fn set_fun_version_all(&mut self, version: FunVersion) -> Result<()> {
        let [
            controller_0,
            controller_1,
            controller_2,
            controller_3,
            controller_4,
            controller_5,
        ] = controllers_as_array(&mut self.controllers);
        tokio::try_join!(
            controller_0.set_fun_version(version),
            controller_1.set_fun_version(version),
            controller_2.set_fun_version(version),
            controller_3.set_fun_version(version),
            controller_4.set_fun_version(version),
            controller_5.set_fun_version(version),
        )?;
        Ok(())
    }

    async fn set_all_timeouts(&mut self, timeout: Duration) -> Result<()> {
        let [
            controller_0,
            controller_1,
            controller_2,
            controller_3,
            controller_4,
            controller_5,
        ] = controllers_as_array(&mut self.controllers);
        tokio::try_join!(
            controller_0.set_timeout(timeout),
            controller_1.set_timeout(timeout),
            controller_2.set_timeout(timeout),
            controller_3.set_timeout(timeout),
            controller_4.set_timeout(timeout),
            controller_5.set_timeout(timeout),
        )?;
        Ok(())
    }

    async fn stop_all(&mut self) -> Result<()> {
        let [
            controller_0,
            controller_1,
            controller_2,
            controller_3,
            controller_4,
            controller_5,
        ] = controllers_as_array(&mut self.controllers);
        tokio::try_join!(
            controller_0.stop(),
            controller_1.stop(),
            controller_2.stop(),
            controller_3.stop(),
            controller_4.stop(),
            controller_5.stop(),
        )?;
        Ok(())
    }
}

fn controllers_as_array(
    controllers: &mut [MotorController],
) -> [&mut MotorController; SERIAL_PORT_COUNT] {
    let controllers: &mut [MotorController; SERIAL_PORT_COUNT] = controllers
        .try_into()
        .expect("Mini Pi Plus has exactly six motor controllers");
    let [
        controller_0,
        controller_1,
        controller_2,
        controller_3,
        controller_4,
        controller_5,
    ] = controllers;
    [
        controller_0,
        controller_1,
        controller_2,
        controller_3,
        controller_4,
        controller_5,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn topology_has_unique_joint_names_and_two_unattached_wrist_slots() {
        let mut names = HashMap::new();
        for joint in JOINT_DEFINITIONS {
            assert!(names.insert(joint.name, ()).is_none());
        }
        assert_eq!(
            JOINT_DEFINITIONS
                .iter()
                .filter(|joint| joint.attached)
                .count(),
            23
        );
        assert_eq!(
            JOINT_DEFINITIONS
                .iter()
                .filter(|joint| !joint.attached)
                .count(),
            2
        );
        assert!(
            !JOINT_DEFINITIONS
                .iter()
                .find(|joint| joint.name == "l_wrist_joint")
                .unwrap()
                .attached
        );
        assert!(
            !JOINT_DEFINITIONS
                .iter()
                .find(|joint| joint.name == "r_wrist_joint")
                .unwrap()
                .attached
        );
    }

    #[test]
    fn topology_preserves_reversal_and_home_offsets() {
        let left_calf = JOINT_DEFINITIONS
            .iter()
            .find(|joint| joint.name == "l_calf_joint")
            .unwrap();
        assert!(left_calf.motor.reverse);
        let left_shoulder_roll = JOINT_DEFINITIONS
            .iter()
            .find(|joint| joint.name == "l_shoulder_roll_joint")
            .unwrap();
        assert_eq!(
            left_shoulder_roll.motor.offset_rad,
            -std::f32::consts::FRAC_PI_2
        );
    }

    #[test]
    fn topology_preserves_known_good_offsets_and_joint_specific_gains() {
        let joint = |name| {
            JOINT_DEFINITIONS
                .iter()
                .find(|joint| joint.name == name)
                .unwrap()
        };
        assert_eq!(
            joint("l_shoulder_pitch_joint").motor.offset_rad,
            std::f32::consts::FRAC_PI_2
        );
        assert_eq!(
            joint("r_shoulder_pitch_joint").motor.offset_rad,
            std::f32::consts::FRAC_PI_2
        );
        assert_eq!(joint("l_elbow_joint").motor.offset_rad, 0.0);
        assert_eq!(joint("r_elbow_joint").motor.offset_rad, 0.0);
        assert_eq!(joint("l_ankle_pitch_joint").proportional_gain, 210.0);
        assert_eq!(joint("l_hip_roll_joint").derivative_gain, 1.4);
        assert_eq!(joint("l_shoulder_pitch_joint").proportional_gain, 6.0);
        assert_eq!(
            joint("head_yaw_joint").proportional_gain,
            HEAD_POSITION_PROPORTIONAL_GAIN
        );
        assert_eq!(
            joint("head_yaw_joint").derivative_gain,
            HEAD_POSITION_DERIVATIVE_GAIN
        );
        assert_eq!(
            joint("head_pitch_joint").proportional_gain,
            HEAD_POSITION_PROPORTIONAL_GAIN
        );
        assert_eq!(
            joint("head_pitch_joint").derivative_gain,
            HEAD_POSITION_DERIVATIVE_GAIN
        );
        assert_eq!(
            joint("waist_yaw_joint").proportional_gain,
            WAIST_POSITION_PROPORTIONAL_GAIN
        );
        assert_eq!(
            joint("waist_yaw_joint").derivative_gain,
            WAIST_POSITION_DERIVATIVE_GAIN
        );
    }
}

#[cfg(all(test, unix))]
mod communication_tests {
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::time::timeout;
    use tokio_serial::SerialStream;

    use super::*;
    use crate::protocol::{CommandId, FrameReader, encode_frame};
    use crate::yesense_protocol::{DATA_ID_ACCELERATION, encode_frame as encode_sensor_frame};

    fn test_robot(period: Duration) -> (MiniPiPlus, Vec<SerialStream>, SerialStream) {
        let mut controllers = Vec::with_capacity(SERIAL_PORT_COUNT);
        let mut peers = Vec::with_capacity(SERIAL_PORT_COUNT);
        for port_index in 0..SERIAL_PORT_COUNT {
            let (peer, port) = SerialStream::pair().unwrap();
            let motors = JOINT_DEFINITIONS
                .iter()
                .filter(|joint| joint.port_index == port_index)
                .map(|joint| joint.motor)
                .collect();
            controllers.push(MotorController::from_port(port, motors));
            peers.push(peer);
        }
        let (sensor_peer, sensor_port) = SerialStream::pair().unwrap();
        let driver =
            BackgroundDriver::start(controllers, YesenseDriver::from_port(sensor_port), period);
        (
            MiniPiPlus {
                driver: Arc::new(driver),
            },
            peers,
            sensor_peer,
        )
    }

    fn motor_on_port(robot: &MiniPiPlus, port_index: usize) -> MiniPiPlusMotor {
        let joint = JOINT_DEFINITIONS
            .iter()
            .find(|joint| joint.attached && joint.port_index == port_index)
            .unwrap();
        robot.motor(joint.name).unwrap()
    }

    fn reply(motor_id: u8, position: i16) -> Vec<u8> {
        let mut payload = vec![motor_id, 0, 0];
        payload.extend_from_slice(&position.to_le_bytes());
        payload.extend_from_slice(&100_i16.to_le_bytes());
        payload.extend_from_slice(&200_i16.to_le_bytes());
        let mut bytes = Vec::new();
        encode_frame(CommandId::MotorState2, &payload, &mut bytes).unwrap();
        bytes
    }

    fn immediately_ready<T>(future: impl Future<Output = T>) -> T {
        let mut future = pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("cached state reads must complete on their first poll"),
        }
    }

    async fn wait_for_reply(motor: &MiniPiPlusMotor) {
        let mut updates = motor.driver.boards[motor.port_index].state_updates(motor.motor.id);
        timeout(Duration::from_secs(2), async {
            while updates.borrow().is_none() {
                updates.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
    }

    async fn receive_command(
        peer: &mut SerialStream,
        command: CommandId,
    ) -> crate::protocol::DecodedFrame {
        timeout(Duration::from_secs(2), async {
            let mut reader = FrameReader::new();
            let mut buffer = [0_u8; 512];
            loop {
                while let Some(frame) = reader.next_frame() {
                    if frame.command == command {
                        return frame;
                    }
                }
                let received = peer.read(&mut buffer).await.unwrap();
                assert!(received > 0);
                reader.extend(&buffer[..received]);
            }
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn fragmented_replies_publish_without_waiting_for_other_boards() {
        let (mut robot, mut peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let mut motor = motor_on_port(&robot, 0);
        let mut idle_motor = motor_on_port(&robot, 1);
        assert!(matches!(
            immediately_ready(motor.read_state()),
            Err(Error::TelemetryUnavailable(_))
        ));
        assert_eq!(motor.latest_state_received_at(), None);
        let frame = reply(motor.motor.id, 1000);
        peers[0].write_all(&frame[..5]).await.unwrap();
        tokio::task::yield_now().await;
        assert_eq!(motor.latest_state_received_at(), None);
        peers[0].write_all(&frame[5..]).await.unwrap();
        wait_for_reply(&motor).await;
        let state = immediately_ready(motor.read_state()).unwrap();
        let received_at = motor.latest_state_received_at().unwrap();
        let expected_position = crate::motor::position_from_i16(1000);
        let expected_position = if motor.motor.reverse {
            -expected_position
        } else {
            expected_position
        } - JointAngle::new::<radian>(motor.motor.offset_rad);
        assert_eq!(
            state.position.get::<radian>(),
            f64::from(expected_position.get::<radian>())
        );
        assert!(matches!(
            immediately_ready(idle_motor.read_state()),
            Err(Error::TelemetryUnavailable(_))
        ));
        assert_eq!(immediately_ready(motor.read_state()).unwrap(), state);
        assert_eq!(motor.latest_state_received_at(), Some(received_at));
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn fresh_motor_request_returns_new_state_with_its_receive_time() {
        let (mut robot, mut peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let motor = motor_on_port(&robot, 0);
        peers[0]
            .write_all(&reply(motor.motor.id, 1000))
            .await
            .unwrap();
        wait_for_reply(&motor).await;
        let old_received_at = motor.latest_state_received_at().unwrap();
        let required_motors = [motor.joint_name];
        let (result, ()) = tokio::join!(
            robot.read_fresh_motor_states(&required_motors, Duration::from_millis(100)),
            async {
                tokio::time::sleep(Duration::from_millis(5)).await;
                peers[0]
                    .write_all(&reply(motor.motor.id, 3000))
                    .await
                    .unwrap();
            }
        );
        let snapshots = result.unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(
            snapshots[0].state.position.get::<radian>(),
            crate::motor::position_from_i16(3000).get::<radian>()
        );
        assert!(snapshots[0].received_at > old_received_at);
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn corrupt_and_unrelated_replies_preserve_cache_until_valid_update() {
        let (mut robot, mut peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let mut motor = motor_on_port(&robot, 0);
        peers[0]
            .write_all(&reply(motor.motor.id, 1000))
            .await
            .unwrap();
        wait_for_reply(&motor).await;
        let original = immediately_ready(motor.read_state()).unwrap();
        let original_time = motor.latest_state_received_at();
        let mut corrupt = reply(motor.motor.id, 2000);
        *corrupt.last_mut().unwrap() ^= 0xFF;
        corrupt.extend_from_slice(&reply(127, 2000));
        peers[0].write_all(&corrupt).await.unwrap();
        tokio::task::yield_now().await;
        assert_eq!(immediately_ready(motor.read_state()).unwrap(), original);
        assert_eq!(motor.latest_state_received_at(), original_time);
        let mut updates = motor.driver.boards[0].state_updates(motor.motor.id);
        updates.borrow_and_update();
        peers[0]
            .write_all(&reply(motor.motor.id, 3000))
            .await
            .unwrap();
        timeout(Duration::from_secs(2), updates.changed())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(immediately_ready(motor.read_state()).unwrap(), original);
        assert!(motor.latest_state_received_at() > original_time);
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn motor_and_robot_operations_run_in_spawned_tasks() {
        let (mut robot, mut peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let mut motor = motor_on_port(&robot, 0);
        peers[0]
            .write_all(&reply(motor.motor.id, 1000))
            .await
            .unwrap();
        wait_for_reply(&motor).await;
        let mut other_robot = robot.clone();
        tokio::spawn(async move {
            motor.read_state().await.unwrap();
            motor
                .set_position_target(Angle::new::<radian>(0.2), 1.0, 0.1)
                .await
                .unwrap();
            other_robot.flush_commands().await.unwrap();
            other_robot.read_measurement().await.unwrap();
            other_robot.stop_all().await.unwrap();
        })
        .await
        .unwrap();
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn flush_batches_targets_and_stop_discards_staged_targets() {
        let (mut robot, mut peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let mut motor = motor_on_port(&robot, 0);
        motor
            .set_position_target(Angle::new::<radian>(0.2), 1.0, 0.1)
            .await
            .unwrap();
        robot.flush_commands().await.unwrap();
        let frame = receive_command(&mut peers[0], CommandId::PosVelTorqueKpKd).await;
        assert_eq!(frame.payload.len(), 60);
        let original = frame.payload;
        motor
            .set_position_target(Angle::new::<radian>(0.4), 1.0, 0.1)
            .await
            .unwrap();
        robot.stop_all().await.unwrap();
        receive_command(&mut peers[0], CommandId::Stop).await;
        robot.flush_commands().await.unwrap();
        assert!(
            robot.driver.boards[0]
                .pending
                .lock()
                .unwrap()
                .take_frame()
                .is_none()
        );
        motor
            .set_position_target(Angle::new::<radian>(0.2), 1.0, 0.1)
            .await
            .unwrap();
        robot.flush_commands().await.unwrap();
        assert_eq!(
            receive_command(&mut peers[0], CommandId::PosVelTorqueKpKd)
                .await
                .payload,
            original
        );
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn sensor_updates_and_disconnects_are_visible_to_cached_reads() {
        let (mut robot, _peers, mut sensor_peer) = test_robot(Duration::from_secs(60));
        let values = [1.0_f32, 2.0, 3.0];
        let mut payload = vec![DATA_ID_ACCELERATION, 12];
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let mut updates = robot.driver.measurement_updates();
        sensor_peer
            .write_all(&encode_sensor_frame(1, &payload))
            .await
            .unwrap();
        timeout(Duration::from_secs(2), updates.changed())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(
            immediately_ready(robot.read_measurement())
                .unwrap()
                .acceleration[0]
                .get::<uom::si::acceleration::meter_per_second_squared>(),
            0.0
        );
        drop(sensor_peer);
        timeout(Duration::from_secs(2), async {
            loop {
                if matches!(
                    immediately_ready(robot.read_measurement()),
                    Err(Error::CommunicationFailed(_))
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(robot.shutdown().await.is_err());
    }

    #[tokio::test]
    async fn shutdown_joins_workers_and_invalidates_cloned_handles() {
        let (mut robot, _peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let mut motor = motor_on_port(&robot, 0);
        robot.shutdown().await.unwrap();
        assert!(robot.driver.tasks_finished());
        assert!(matches!(
            immediately_ready(motor.read_state()),
            Err(Error::CommunicationStopped)
        ));
        assert!(matches!(
            motor
                .set_position_target(Angle::new::<radian>(0.0), 1.0, 0.1)
                .await,
            Err(Error::CommunicationStopped)
        ));
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropping_final_owner_cancels_workers() {
        let (robot, _peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let tasks = robot.driver.task_abort_handles();
        let motor = motor_on_port(&robot, 0);
        drop(robot);
        assert!(tasks.iter().all(|task| !task.is_finished()));
        drop(motor);
        timeout(Duration::from_secs(2), async {
            while tasks.iter().any(|task| !task.is_finished()) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn zero_request_period_is_rejected_before_opening_hardware() {
        let configuration = MiniPiPlusConfig {
            motor_telemetry_period: std::time::Duration::from_millis(0),
            ..MiniPiPlusConfig::default()
        };
        assert!(matches!(
            MiniPiPlus::open(configuration).await,
            Err(Error::InvalidTelemetryPeriod)
        ));
    }
    #[tokio::test]
    async fn disconnected_motor_board_reports_failure_without_affecting_other_boards() {
        let (mut robot, mut peers, _sensor_peer) = test_robot(Duration::from_secs(60));
        let mut motor = motor_on_port(&robot, 0);
        let mut other_motor = motor_on_port(&robot, 1);
        peers[1]
            .write_all(&reply(other_motor.motor.id, 1000))
            .await
            .unwrap();
        wait_for_reply(&other_motor).await;
        let other_state = immediately_ready(other_motor.read_state()).unwrap();
        drop(peers.remove(0));
        timeout(Duration::from_secs(2), async {
            loop {
                if matches!(
                    immediately_ready(motor.read_state()),
                    Err(Error::CommunicationFailed(_))
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            immediately_ready(other_motor.read_state()).unwrap(),
            other_state
        );
        assert!(robot.shutdown().await.is_err());
        assert!(robot.driver.tasks_finished());
    }
}
