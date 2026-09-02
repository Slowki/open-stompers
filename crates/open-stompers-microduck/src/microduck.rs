//! Microduck motor and inertial measurement hardware.
//!
//! Opening a robot starts background telemetry on the shared Dynamixel bus.
//! Clone robot and motor handles into Tokio tasks; hardware trait reads return
//! cached values immediately. Motor telemetry is unavailable before the first
//! reply. Position commands are staged until [`Microduck::flush_commands`].
//! Await [`Microduck::shutdown`] to disable torque and join the bus task.
//!
//! XL330 servos must already be in position mode, with the knee `homing_offset`
//! values stored in EEPROM. `kp` and `kd` are position P and D gain registers,
//! not Newton-metre gains: the control loop uses `200` and `0`. The IMU
//! acceleration is specific force estimated from the board's fused orientation,
//! because the 12-byte bus block carries gyro and quaternion rather than raw
//! accelerometer counts.
//!
//! ```no_run
//! use open_stompers_hardware::Motor;
//! use open_stompers_microduck::{Error, Microduck, MicroduckConfig};
//!
//! # async fn run() -> open_stompers_microduck::Result<()> {
//! let mut robot = Microduck::open(MicroduckConfig::default()).await?;
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

use std::sync::Arc;
use std::time::Instant;

use tokio_serial::SerialPortBuilderExt;

use open_stompers_hardware::{
    InertialMeasurement, InertialMeasurementSensor, Motor, MotorState, PositionControlledMotor,
};
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{Angle, AngularVelocity, Torque};
use uom::si::torque::newton_meter;

use crate::background::{BackgroundDriver, BusSettings};
use crate::config::MicroduckConfig;
use crate::error::{Error, Result};
use crate::model::JOINT_DEFINITIONS;
use crate::protocol::POSITION_GAIN_MAX;

/// Microduck robot coordinator for batched motor and IMU I/O.
#[derive(Clone)]
pub struct Microduck {
    driver: Arc<BackgroundDriver>,
}

/// Handle for one Microduck joint servo.
#[derive(Clone)]
pub struct MicroduckMotor {
    driver: Arc<BackgroundDriver>,
    joint_name: &'static str,
    motor_id: u8,
}

impl Microduck {
    /// Open the Dynamixel bus and start background telemetry.
    ///
    /// # Errors
    ///
    /// Returns an error if the telemetry period or response timeout is zero, or
    /// if the serial port cannot be opened.
    pub async fn open(configuration: MicroduckConfig) -> Result<Self> {
        if configuration.motor_telemetry_period.is_zero() {
            return Err(Error::InvalidTelemetryPeriod);
        }
        if configuration.response_timeout.is_zero() {
            return Err(Error::InvalidResponseTimeout);
        }
        let port = tokio_serial::new(&configuration.serial_path, configuration.baudrate)
            .exclusive(true)
            .open_native_async()
            .map_err(|source| Error::OpenPort {
                path: configuration.serial_path.clone(),
                source,
            })?;
        Ok(Self::from_port(
            port,
            BusSettings {
                telemetry_period: configuration.motor_telemetry_period,
                response_timeout: configuration.response_timeout,
                fast_sync_read: configuration.fast_sync_read,
                mount: configuration.imu_mount,
            },
        ))
    }

    pub(crate) fn from_port(port: tokio_serial::SerialStream, settings: BusSettings) -> Self {
        Self {
            driver: Arc::new(BackgroundDriver::start(port, settings)),
        }
    }

    /// Return a handle for a joint servo by URDF name.
    #[must_use]
    pub fn motor(&self, joint_name: &str) -> Option<MicroduckMotor> {
        let definition = JOINT_DEFINITIONS
            .iter()
            .find(|joint| joint.name == joint_name)?;
        Some(MicroduckMotor {
            driver: Arc::clone(&self.driver),
            joint_name: definition.name,
            motor_id: definition.id,
        })
    }

    /// List joint names exposed as motors, including the mouth.
    pub fn motor_names(&self) -> impl Iterator<Item = &'static str> {
        JOINT_DEFINITIONS.iter().map(|joint| joint.name)
    }

    /// Receive time of the latest IMU sample, or `None` before the first reply.
    #[must_use]
    pub fn latest_measurement_received_at(&self) -> Option<Instant> {
        self.driver.imu_snapshot().received_at
    }

    /// Latest trunk-to-world quaternion, scalar-first `[w, x, y, z]`.
    ///
    /// Returns `None` until the IMU board has produced a quaternion. A later
    /// all-zero quaternion block holds this value rather than reporting identity.
    #[must_use]
    pub fn latest_orientation_wxyz(&self) -> Option<[f64; 4]> {
        self.driver.imu_snapshot().orientation_wxyz
    }

    /// Write staged goal positions, enabling torque on the commanded servos.
    ///
    /// Gains are written only when they differ from the last flushed values.
    ///
    /// # Errors
    ///
    /// Returns an error if communication has stopped or the bus write fails.
    pub async fn flush_commands(&mut self) -> Result<()> {
        self.driver.flush().await
    }

    /// Disable torque on every servo and discard staged commands.
    ///
    /// # Errors
    ///
    /// Returns an error if communication has stopped or the bus write fails.
    pub async fn stop_all(&mut self) -> Result<()> {
        self.driver.stop_all().await
    }

    /// Disable torque, close the bus task, and wait for it to finish.
    ///
    /// Shut down every clone and motor handle sharing this connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the final torque-off write fails or the bus task
    /// already failed.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.driver.shutdown().await
    }
}

impl MicroduckMotor {
    /// Receive time of the latest valid reply for this servo.
    ///
    /// Returns `None` before the first reply. The timestamp is preserved across
    /// reply gaps and after shutdown; check [`Motor::read_state`] for connection
    /// errors.
    #[must_use]
    pub fn latest_state_received_at(&self) -> Option<Instant> {
        self.driver
            .snapshot(self.motor_id)
            .map(|snapshot| snapshot.received_at)
    }
}

impl InertialMeasurementSensor for Microduck {
    type Error = Error;

    async fn read_measurement(&mut self) -> Result<InertialMeasurement> {
        self.driver.check_open()?;
        Ok(self.driver.imu_snapshot().measurement)
    }
}

impl Motor for MicroduckMotor {
    type Error = Error;

    async fn read_state(&mut self) -> Result<MotorState> {
        self.driver.check_open()?;
        let state = self
            .driver
            .snapshot(self.motor_id)
            .ok_or_else(|| Error::TelemetryUnavailable(self.joint_name.to_owned()))?;
        Ok(MotorState {
            position: Angle::new::<radian>(state.position_rad),
            velocity: AngularVelocity::new::<radian_per_second>(state.velocity_rad_per_second),
            torque: Some(Torque::new::<newton_meter>(state.torque_newton_meters)),
        })
    }
}

impl PositionControlledMotor for MicroduckMotor {
    async fn set_position_target(&mut self, target: Angle, kp: f64, kd: f64) -> Result<()> {
        let position = target.get::<radian>();
        if !position.is_finite() || !kp.is_finite() || !kd.is_finite() || kp < 0.0 || kd < 0.0 {
            return Err(Error::InvalidCommandValue(self.joint_name.to_owned()));
        }
        let proportional = gain_register(kp)
            .ok_or_else(|| Error::InvalidCommandValue(self.joint_name.to_owned()))?;
        let derivative = gain_register(kd)
            .ok_or_else(|| Error::InvalidCommandValue(self.joint_name.to_owned()))?;
        self.driver.check_open()?;
        self.driver.stage(
            self.motor_id,
            crate::model::position_to_raw(target),
            proportional,
            derivative,
        );
        Ok(())
    }
}

fn gain_register(gain: f64) -> Option<u16> {
    if !(0.0..=f64::from(POSITION_GAIN_MAX)).contains(&gain) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(gain.round() as u16)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn every_joint_can_be_addressed_by_name() {
        let names = JOINT_DEFINITIONS
            .iter()
            .map(|joint| joint.name)
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 15);
        assert!(names.contains(&"mouth"));
        assert!(names.contains(&"left_knee"));
    }

    #[tokio::test]
    async fn zero_periods_are_rejected_before_opening_hardware() {
        let configuration = MicroduckConfig {
            motor_telemetry_period: Duration::ZERO,
            ..MicroduckConfig::default()
        };
        assert!(matches!(
            Microduck::open(configuration).await,
            Err(Error::InvalidTelemetryPeriod)
        ));
        let configuration = MicroduckConfig {
            response_timeout: Duration::ZERO,
            ..MicroduckConfig::default()
        };
        assert!(matches!(
            Microduck::open(configuration).await,
            Err(Error::InvalidResponseTimeout)
        ));
    }
}

#[cfg(all(test, unix))]
mod communication_tests {
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::time::timeout;
    use tokio_serial::SerialStream;
    use uom::si::acceleration::meter_per_second_squared;
    use uom::si::angular_velocity::radian_per_second;

    use super::*;
    use crate::background::BusSettings;
    use crate::imu::ImuMount;
    use crate::model::IMU_ID;
    use crate::protocol::{
        BROADCAST_ID, INSTRUCTION_FAST_SYNC_READ, INSTRUCTION_SYNC_WRITE, PRESENT_BLOCK_LEN,
        PacketReader, encode_fast_sync_status, instruction_params,
    };

    fn test_settings() -> BusSettings {
        BusSettings {
            telemetry_period: Duration::from_millis(20),
            response_timeout: Duration::from_secs(2),
            fast_sync_read: true,
            mount: ImuMount::Beta,
        }
    }

    fn test_robot() -> (Microduck, SerialStream) {
        let (peer, port) = SerialStream::pair().unwrap();
        (Microduck::from_port(port, test_settings()), peer)
    }

    fn immediately_ready<T>(future: impl Future<Output = T>) -> T {
        let mut future = pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("cached reads must complete on their first poll"),
        }
    }

    fn motor_block(position: i32, velocity: i32, current: i16) -> [u8; 12] {
        let mut block = [0_u8; 12];
        block[2..4].copy_from_slice(&current.to_le_bytes());
        block[4..8].copy_from_slice(&velocity.to_le_bytes());
        block[8..12].copy_from_slice(&position.to_le_bytes());
        block
    }

    fn imu_block(gyro_x: i16) -> [u8; 12] {
        let mut block = [0_u8; 12];
        block[..2].copy_from_slice(&gyro_x.to_le_bytes());
        block[6..8].copy_from_slice(&0x2E66_u16.to_le_bytes());
        block
    }

    async fn answer_reads(mut peer: SerialStream, gyro_x: i16, position: i32) {
        let mut reader = PacketReader::new();
        let mut buffer = [0_u8; 512];
        let motor = motor_block(position, 10, 1_000);
        let imu = imu_block(gyro_x);
        loop {
            let received = peer.read(&mut buffer).await.unwrap();
            if received == 0 {
                break;
            }
            reader.push(&buffer[..received]);
            while let Some(packet) = reader.next_packet() {
                if packet.instruction != INSTRUCTION_FAST_SYNC_READ {
                    continue;
                }
                let params = instruction_params(&packet);
                let ids = &params[4..];
                let blocks = ids
                    .iter()
                    .map(|id| {
                        if *id == IMU_ID {
                            imu.to_vec()
                        } else {
                            motor.to_vec()
                        }
                    })
                    .collect::<Vec<_>>();
                let references = ids
                    .iter()
                    .zip(&blocks)
                    .map(|(id, block)| (*id, block.as_slice()))
                    .collect::<Vec<_>>();
                let response = encode_fast_sync_status(&references);
                assert_eq!(PRESENT_BLOCK_LEN, 12);
                peer.write_all(&response).await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn telemetry_publishes_motor_state_and_imu_without_blocking() {
        let (mut robot, peer) = test_robot();
        let answer = tokio::spawn(answer_reads(peer, 1_000, 2048));
        let mut motor = robot.motor("left_hip_yaw").unwrap();
        assert!(matches!(
            immediately_ready(motor.read_state()),
            Err(Error::TelemetryUnavailable(_))
        ));
        let mut updates = motor.driver.state_updates(motor.motor_id);
        timeout(Duration::from_secs(2), async {
            while updates.borrow().is_none() {
                updates.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        let state = immediately_ready(motor.read_state()).unwrap();
        let received_at = motor.latest_state_received_at().unwrap();
        assert!(state.position.get::<radian>().abs() < 1e-9);
        assert!(state.velocity.get::<radian_per_second>() > 0.0);
        let torque = state.torque.unwrap().get::<newton_meter>();
        assert!((torque - 0.354).abs() < 1e-9);
        let measurement = immediately_ready(robot.read_measurement()).unwrap();
        assert!(measurement.angular_velocity[0].get::<radian_per_second>() > 0.0);
        assert!(measurement.acceleration[2].get::<meter_per_second_squared>() > 9.0);
        assert!(robot.latest_orientation_wxyz().is_some());
        assert_eq!(robot.latest_measurement_received_at(), Some(received_at));
        assert_eq!(immediately_ready(motor.read_state()).unwrap(), state);
        robot.shutdown().await.unwrap();
        answer.abort();
    }

    #[tokio::test]
    async fn flush_enables_torque_and_writes_the_staged_goal() {
        let (mut robot, mut peer) = test_robot();
        let mut motor = robot.motor("right_knee").unwrap();
        motor
            .set_position_target(Angle::new::<radian>(0.0), 200.0, 0.0)
            .await
            .unwrap();
        let flush = tokio::spawn(async move {
            robot.flush_commands().await.unwrap();
            robot
        });
        let mut reader = PacketReader::new();
        let mut buffer = [0_u8; 512];
        let mut saw_torque = false;
        let mut saw_goal = false;
        timeout(Duration::from_secs(2), async {
            while !saw_torque || !saw_goal {
                let received = peer.read(&mut buffer).await.unwrap();
                reader.push(&buffer[..received]);
                while let Some(packet) = reader.next_packet() {
                    if packet.id == BROADCAST_ID && packet.instruction == INSTRUCTION_FAST_SYNC_READ
                    {
                        continue;
                    }
                    assert_eq!(packet.instruction, INSTRUCTION_SYNC_WRITE);
                    let params = instruction_params(&packet);
                    let address = u16::from_le_bytes([params[0], params[1]]);
                    if address == 64 {
                        saw_torque = true;
                        assert_eq!(params[4], motor.motor_id);
                        assert_eq!(params[5], 1);
                    }
                    if address == 116 {
                        saw_goal = true;
                        assert_eq!(params[4], motor.motor_id);
                        assert_eq!(&params[5..9], &2048_i32.to_le_bytes());
                    }
                }
            }
        })
        .await
        .unwrap();
        let mut robot = flush.await.unwrap();
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_disables_torque_and_invalidates_cloned_handles() {
        let (mut robot, _peer) = test_robot();
        let mut motor = robot.motor("mouth").unwrap();
        robot.shutdown().await.unwrap();
        assert!(robot.driver.tasks_finished());
        assert!(matches!(
            immediately_ready(motor.read_state()),
            Err(Error::CommunicationStopped)
        ));
        assert!(matches!(
            motor
                .set_position_target(Angle::new::<radian>(0.0), 1.0, 0.0)
                .await,
            Err(Error::CommunicationStopped)
        ));
        robot.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropping_the_final_owner_cancels_the_bus_task() {
        let (robot, _peer) = test_robot();
        let tasks = robot.driver.task_abort_handles();
        let motor = robot.motor("head_yaw").unwrap();
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
    async fn a_disconnected_bus_fails_cached_reads() {
        let (mut robot, peer) = test_robot();
        let mut motor = robot.motor("left_ankle").unwrap();
        drop(peer);
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
        assert!(robot.shutdown().await.is_err());
        assert!(robot.driver.tasks_finished());
    }
}
