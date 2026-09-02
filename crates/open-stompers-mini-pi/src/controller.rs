//! Asynchronous driver for a High Torque motor CDC-serial communications board.
//!
//! A [`MotorController`] owns one serial port (typically a `/dev/ttyACM*`
//! device exposed by the robot's communication board) and addresses every
//! motor attached to that port by its 1-based `id`.
//!
//! # Example
//!
//! ```no_run
//! use open_stompers_mini_pi::{MotorConfig, MotorController, MotorType};
//! use uom::si::angle::radian;
//! use uom::si::f32::Angle;
//!
//! # async fn run() -> open_stompers_mini_pi::Result<()> {
//! let motors = vec![
//!     MotorConfig::new(1, MotorType::M5047_36_2),
//!     MotorConfig::new(2, MotorType::M5047_36_2),
//! ];
//! let mut controller = MotorController::open("/dev/ttyACM0", 4_000_000, motors)?;
//!
//! loop {
//!     controller.request_motor_state().await?;
//!     controller.poll().await?;
//!     controller.set_position(1, Angle::new::<radian>(0.0))?;
//!     controller.set_position(2, Angle::new::<radian>(0.25))?;
//!     controller.flush().await?;
//! }
//! # }
//! ```

use std::collections::HashMap;
use std::io;
use std::time::Duration;

use serialport::{ClearBuffer, SerialPort};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{sleep, timeout};
use tokio_serial::{SerialPortBuilderExt, SerialStream};
use uom::si::angle::radian;
use uom::si::f32::{Angle, AngularVelocity, Torque};

use crate::error::{Error, Result};
use crate::motor::{
    MotorConfig, MotorState, pid_gain_to_i16, position_from_i16, position_to_i16, torque_from_i16,
    torque_to_i16, velocity_from_i16, velocity_to_i16,
};
use crate::protocol::{CommandId, DecodedFrame, FrameReader, encode_frame};

/// Entry size (bytes) of the payload for `MODE_POSITION`.
const POSITION_ENTRY_SIZE: usize = 2;

/// Entry size (bytes) of the payload for `MODE_POS_VEL_TQE` (`pos`, `val`,
/// `tqe`, each `i16`).
const POS_VEL_MAX_TORQUE_ENTRY_SIZE: usize = 6;

/// Entry size (bytes) of the payload for `MODE_POS_VEL_TQE_KP_KD` (`pos`,
/// `val`, `tqe`, `rkp`, `rkd`, each `i16`). `MODE_POS_VEL_TQE_KP_KD2` uses the
/// same layout.
const POS_VEL_TORQUE_KP_KD_ENTRY_SIZE: usize = 10;

/// Entry size (bytes) of the RX payload for `MODE_MOTOR_STATE`
/// (`id`, `pos`, `val`, `tqe`).
const MOTOR_STATE_ENTRY_SIZE: usize = 7;

/// Entry size (bytes) of the RX payload for `MODE_MOTOR_STATE2`
/// (`id`, `mode`, `fault`, `pos`, `val`, `tqe`).
const MOTOR_STATE2_ENTRY_SIZE: usize = 9;

/// Broadcast addressing byte used by the reference SDK for 1-byte control
/// frames such as `MODE_STOP`, `MODE_RESET`, `MODE_MOTOR_STATE` request, etc.
const BROADCAST_ADDRESS: u8 = 0x7F;

/// Maximum wait for a standalone response poll.
const DEFAULT_SERIAL_TIMEOUT: Duration = Duration::from_millis(50);

/// Sentinel used for "do not update this motor's position" slots.
const POSITION_SENTINEL_BYTES: [u8; 2] = (-0x8000i16).to_le_bytes();

/// Protocol feature version values recognized by the comm board's `MODE_FUN_V` command. `FunVersion::V5` is the newest
/// on-device protocol used by Mini Pi Plus motors running firmware ≥ 4.4.6.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FunVersion {
    /// 1.0
    V1 = 1,
    /// 2.0
    V2 = 2,
    /// 3.0
    V3 = 3,
    /// 4.0
    V4 = 4,
    /// 5.0
    V5 = 5,
}

impl FunVersion {
    /// Decode a raw comm-board reply byte into a known protocol version.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            4 => Some(Self::V4),
            5 => Some(Self::V5),
            _ => None,
        }
    }
}

/// Number of times `set_motor_count` / `set_fun_version` resend the request while waiting for the comm board to reply.
const HANDSHAKE_MAX_ATTEMPTS: u32 = 50;

/// Maximum wait before retrying a handshake request.
const HANDSHAKE_REPLY_TIMEOUT: Duration = Duration::from_millis(20);

/// Staged transmit payload for the currently selected command.
///
/// The reference SDK keeps exactly one outgoing frame per CAN port in flight
/// and rebuilds its payload only when the command type changes, which is what
/// this struct models.
pub(crate) struct PendingTx {
    command: Option<CommandId>,
    payload: Vec<u8>,
    id_max: u8,
}

impl PendingTx {
    const fn new(id_max: u8) -> Self {
        Self {
            command: None,
            payload: Vec::new(),
            id_max,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.command = None;
        self.payload.clear();
    }

    pub(crate) fn take_frame(&mut self) -> Option<(CommandId, Vec<u8>)> {
        let command = self.command.take()?;
        Some((command, std::mem::take(&mut self.payload)))
    }

    fn slot_index(motor_id: u8, entry_size: usize) -> usize {
        usize::from(motor_id - 1) * entry_size
    }

    /// Ensure the staged payload matches `command`, resetting all slots to
    /// the provided sentinel bytes when switching.
    fn ensure_command(&mut self, command: CommandId, entry_size: usize, sentinel: &[u8]) {
        debug_assert_eq!(sentinel.len(), entry_size);
        if self.command == Some(command) {
            return;
        }
        self.command = Some(command);
        let total_len = usize::from(self.id_max) * entry_size;
        self.payload.clear();
        self.payload.resize(total_len, 0);
        for chunk in self.payload.chunks_exact_mut(entry_size) {
            chunk.copy_from_slice(sentinel);
        }
    }

    /// Stage `MODE_POSITION` for a single motor. Other slots stay untouched
    /// (either the sentinel from the most recent `ensure_command`, or
    /// whatever value was staged earlier this cycle).
    fn stage_position(&mut self, motor: MotorConfig, position: Angle) {
        self.ensure_command(
            CommandId::Position,
            POSITION_ENTRY_SIZE,
            &POSITION_SENTINEL_BYTES,
        );
        let adjusted = apply_tx_position(motor, position);
        let raw = position_to_i16(adjusted).to_le_bytes();
        let start = Self::slot_index(motor.id, POSITION_ENTRY_SIZE);
        self.payload[start..start + POSITION_ENTRY_SIZE].copy_from_slice(&raw);
    }

    /// Stage `MODE_POS_VEL_TQE` for a single motor (target position with a
    /// maximum velocity and maximum torque).
    fn stage_pos_vel_max_torque(
        &mut self,
        motor: MotorConfig,
        position: Angle,
        max_velocity: AngularVelocity,
        max_torque: Torque,
    ) {
        let mut sentinel = [0u8; POS_VEL_MAX_TORQUE_ENTRY_SIZE];
        sentinel[..2].copy_from_slice(&POSITION_SENTINEL_BYTES);
        self.ensure_command(
            CommandId::PosVelMaxTorque,
            POS_VEL_MAX_TORQUE_ENTRY_SIZE,
            &sentinel,
        );

        // The firmware interprets `vel` and `tqe` as magnitudes (saturation
        // limits on the trajectory), so `reverse` only affects the position.
        let adjusted_position = apply_tx_position(motor, position);
        let velocity_magnitude = max_velocity.abs();
        let torque_magnitude = max_torque.abs();

        let position_raw = position_to_i16(adjusted_position).to_le_bytes();
        let velocity_raw = velocity_to_i16(velocity_magnitude).to_le_bytes();
        let torque_raw = torque_to_i16(torque_magnitude, motor.motor_type).to_le_bytes();

        let start = Self::slot_index(motor.id, POS_VEL_MAX_TORQUE_ENTRY_SIZE);
        self.payload[start..start + 2].copy_from_slice(&position_raw);
        self.payload[start + 2..start + 4].copy_from_slice(&velocity_raw);
        self.payload[start + 4..start + 6].copy_from_slice(&torque_raw);
    }

    /// Stage `MODE_POS_VEL_TQE_KP_KD` for a single motor.
    pub(crate) fn stage_pos_vel_torque_kp_kd(
        &mut self,
        motor: MotorConfig,
        position: Angle,
        velocity: AngularVelocity,
        torque: Torque,
        kp: f32,
        kd: f32,
    ) {
        let mut sentinel = [0u8; POS_VEL_TORQUE_KP_KD_ENTRY_SIZE];
        sentinel[..2].copy_from_slice(&POSITION_SENTINEL_BYTES);
        self.ensure_command(
            CommandId::PosVelTorqueKpKd,
            POS_VEL_TORQUE_KP_KD_ENTRY_SIZE,
            &sentinel,
        );

        let adjusted_position = apply_tx_position(motor, position);
        let scaled_velocity = if motor.reverse { -velocity } else { velocity };
        let scaled_torque = torque;

        let position_raw = position_to_i16(adjusted_position).to_le_bytes();
        let velocity_raw = velocity_to_i16(scaled_velocity).to_le_bytes();
        let torque_raw = torque_to_i16(scaled_torque, motor.motor_type).to_le_bytes();
        let kp_raw = pid_gain_to_i16(kp, motor.motor_type).to_le_bytes();
        let kd_raw = pid_gain_to_i16(kd, motor.motor_type).to_le_bytes();

        let start = Self::slot_index(motor.id, POS_VEL_TORQUE_KP_KD_ENTRY_SIZE);
        self.payload[start..start + 2].copy_from_slice(&position_raw);
        self.payload[start + 2..start + 4].copy_from_slice(&velocity_raw);
        self.payload[start + 4..start + 6].copy_from_slice(&torque_raw);
        self.payload[start + 6..start + 8].copy_from_slice(&kp_raw);
        self.payload[start + 8..start + 10].copy_from_slice(&kd_raw);
    }
}

/// Apply the motor's reverse/offset adjustments for TX-side position.
fn apply_tx_position(motor: MotorConfig, position: Angle) -> Angle {
    let with_offset = position + Angle::new::<radian>(motor.offset_rad);
    if motor.reverse {
        -with_offset
    } else {
        with_offset
    }
}

/// Convert a watchdog duration into the firmware's signed-millisecond encoding.
fn timeout_to_firmware_milliseconds(timeout: Duration) -> Result<i16> {
    i16::try_from(timeout.as_millis()).map_err(|_| Error::TimeoutOutOfRange { timeout })
}

/// Driver for a single High Torque motor CDC-serial port.
pub struct MotorController {
    pub(crate) port: SerialStream,
    motors: HashMap<u8, MotorConfig>,
    pending: PendingTx,
    states: HashMap<u8, MotorState>,
    reader: FrameReader,
    received_at: HashMap<u8, std::time::Instant>,
    tx_scratch: Vec<u8>,
    /// Comm-board firmware version reported in the last `MODE_SET_NUM` reply
    /// (major + minor × 0.1), or `None` if we haven't received one yet.
    port_version: Option<f32>,
    /// Protocol feature version echoed back in the last `MODE_FUN_V` reply.
    fun_version: Option<FunVersion>,
}

impl MotorController {
    /// Open `path` at `baudrate` and register `motors`.
    ///
    /// The highest motor id defines the TX payload length and mirrors the reference SDK's `id_max`.
    pub fn open(path: &str, baudrate: u32, motors: Vec<MotorConfig>) -> Result<Self> {
        let mut port = tokio_serial::new(path, baudrate)
            .open_native_async()
            .map_err(|source| Error::OpenPort {
                path: path.to_string(),
                source,
            })?;

        // Match the defaults of the reference SDK's underlying `serial` library:
        // assert DTR and RTS so the USB-CDC communication board sees the host as
        // "connected" and enables its TX/RX path. serialport-rs leaves these
        // deasserted by default, which can leave writes sitting in the kernel
        // buffer until `poll(POLLOUT)` finally times out.
        let _ = port.write_data_terminal_ready(true);
        let _ = port.write_request_to_send(true);
        // Discard any stale bytes the kernel might have buffered from a prior
        // session before we start framing.
        let _ = port.clear(ClearBuffer::All);

        Ok(Self::from_port(port, motors))
    }

    pub(crate) fn from_port(port: SerialStream, motors: Vec<MotorConfig>) -> Self {
        let id_max = motors.iter().map(|motor| motor.id).max().unwrap_or(0);
        let motor_map = motors
            .into_iter()
            .map(|motor| (motor.id, motor))
            .collect::<HashMap<_, _>>();

        Self {
            port,
            motors: motor_map,
            pending: PendingTx::new(id_max),
            states: HashMap::new(),
            reader: FrameReader::new(),
            received_at: HashMap::new(),
            tx_scratch: Vec::new(),
            port_version: None,
            fun_version: None,
        }
    }

    /// Stage a position command for `motor_id`.
    pub fn set_position(&mut self, motor_id: u8, position: Angle) -> Result<()> {
        let motor = *self
            .motors
            .get(&motor_id)
            .ok_or(Error::UnknownMotorId(motor_id))?;
        self.pending.stage_position(motor, position);
        Ok(())
    }

    /// Stage a combined position/velocity/torque/Kp/Kd command for `motor_id`.
    ///
    /// Sends `MODE_POS_VEL_TQE_KP_KD` (`0x93`), matching HighTorque midware
    /// position control: PD around `position`/`velocity` plus feedforward
    /// `torque`.
    pub fn set_pos_vel_torque_kp_kd(
        &mut self,
        motor_id: u8,
        position: Angle,
        velocity: AngularVelocity,
        torque: Torque,
        kp: f32,
        kd: f32,
    ) -> Result<()> {
        let motor = *self
            .motors
            .get(&motor_id)
            .ok_or(Error::UnknownMotorId(motor_id))?;
        self.pending
            .stage_pos_vel_torque_kp_kd(motor, position, velocity, torque, kp, kd);
        Ok(())
    }

    /// Stage a trajectory command for `motor_id`: drive to `position`
    /// without exceeding `max_velocity` or `max_torque`.
    ///
    /// Mirrors `motor::pos_vel_MAXtqe` in the reference SDK, which sends
    /// `MODE_POS_VEL_TQE` (`0x90`). The motor firmware treats the velocity
    /// and torque fields as magnitude limits, so negative values here are
    /// taken as their absolute value before being scaled onto the wire.
    pub fn set_pos_vel_max_torque(
        &mut self,
        motor_id: u8,
        position: Angle,
        max_velocity: AngularVelocity,
        max_torque: Torque,
    ) -> Result<()> {
        let motor = *self
            .motors
            .get(&motor_id)
            .ok_or(Error::UnknownMotorId(motor_id))?;
        self.pending
            .stage_pos_vel_max_torque(motor, position, max_velocity, max_torque);
        Ok(())
    }

    /// Transmit any pending command frame to the serial port.
    ///
    /// Does nothing when no command has been staged since the last flush.
    pub async fn flush(&mut self) -> Result<()> {
        let Some(command) = self.pending.command else {
            return Ok(());
        };
        self.tx_scratch.clear();
        encode_frame(command, &self.pending.payload, &mut self.tx_scratch)?;
        self.port.write_all(&self.tx_scratch).await?;
        self.pending.reset();
        Ok(())
    }

    /// Send a `MODE_MOTOR_STATE2` broadcast request (the newer, 6-field
    /// telemetry format: `{id, mode, fault, pos, val, tqe}` per motor).
    ///
    /// This matches what the reference SDK sends once it has negotiated
    /// `fun_v >= fun_v4` firmware (the regime Mini Pi Plus runs in with
    /// `control_type: 9`). Responses are decoded by a subsequent call to
    /// [`MotorController::poll`].
    pub async fn request_motor_state(&mut self) -> Result<()> {
        self.send_one_shot(CommandId::MotorState2, &[BROADCAST_ADDRESS])
            .await
    }

    /// Send a `MODE_STOP` broadcast command. Any pending command frame is
    /// dropped (the reference SDK treats stop as a fresh frame).
    pub async fn stop(&mut self) -> Result<()> {
        self.pending.reset();
        self.send_one_shot(CommandId::Stop, &[BROADCAST_ADDRESS])
            .await
    }

    /// Send a `MODE_RESET` broadcast command.
    pub async fn reset(&mut self) -> Result<()> {
        self.send_one_shot(CommandId::Reset, &[BROADCAST_ADDRESS])
            .await
    }

    /// Tell the comm board how many motors live on this CAN bus and wait
    /// for it to reply with its own firmware version.
    ///
    /// Mirrors `canport::set_motor_num` in the reference SDK: resends the
    /// `MODE_SET_NUM` frame until the board replies with a `MODE_SET_NUM`
    /// response that populates [`MotorController::port_version`], or until
    /// we've tried [`HANDSHAKE_MAX_ATTEMPTS`] times.
    ///
    /// Returns the comm board's reported version on success, or `None` if
    /// the board never replied.
    pub async fn set_motor_count(&mut self, motor_count: u8) -> Result<Option<f32>> {
        self.port_version = None;
        for _ in 0..HANDSHAKE_MAX_ATTEMPTS {
            self.send_one_shot(CommandId::SetNum, &[motor_count])
                .await?;
            let deadline = tokio::time::Instant::now() + HANDSHAKE_REPLY_TIMEOUT;
            while self.port_version.is_none() {
                match timeout(
                    deadline.saturating_duration_since(tokio::time::Instant::now()),
                    self.receive_frame(),
                )
                .await
                {
                    Ok(result) => {
                        result?;
                    }
                    Err(_) => break,
                }
            }
            if self.port_version.is_some() {
                break;
            }
        }
        if self.port_version.is_none() {
            tracing::warn!(
                "set_motor_count: no MODE_SET_NUM reply after {HANDSHAKE_MAX_ATTEMPTS} attempts",
            );
        }
        Ok(self.port_version)
    }

    /// Tell the comm board which protocol feature version to speak and wait
    /// for it to echo the selection back.
    ///
    /// Mirrors `canport::set_fun_v` in the reference SDK. For Mini Pi Plus
    /// on newest firmware pass [`FunVersion::V5`].
    pub async fn set_fun_version(&mut self, version: FunVersion) -> Result<Option<FunVersion>> {
        self.fun_version = None;
        for _ in 0..HANDSHAKE_MAX_ATTEMPTS {
            self.send_one_shot(CommandId::FunV, &[version as u8])
                .await?;
            let deadline = tokio::time::Instant::now() + HANDSHAKE_REPLY_TIMEOUT;
            while self.fun_version != Some(version) {
                match timeout(
                    deadline.saturating_duration_since(tokio::time::Instant::now()),
                    self.receive_frame(),
                )
                .await
                {
                    Ok(result) => {
                        result?;
                    }
                    Err(_) => break,
                }
            }
            if self.fun_version == Some(version) {
                break;
            }
        }
        if self.fun_version != Some(version) {
            tracing::warn!(
                "set_fun_version: requested {version:?} but board reported {:?} after {HANDSHAKE_MAX_ATTEMPTS} attempts",
                self.fun_version,
            );
        }
        Ok(self.fun_version)
    }

    /// Latest comm-board firmware version reported via `MODE_SET_NUM`.
    #[must_use]
    pub const fn port_version(&self) -> Option<f32> {
        self.port_version
    }

    /// Latest protocol feature version echoed back by `MODE_FUN_V`.
    #[must_use]
    pub const fn fun_version(&self) -> Option<FunVersion> {
        self.fun_version
    }

    /// Reinitialize the comm board's FDCAN interface.
    ///
    /// Mirrors `canport::canboard_fdcan_reset` in the reference SDK: sends
    /// three back-to-back `MODE_FDCAN_RESET` frames and then sleeps briefly
    /// so the firmware has time to bring the bus back up before the next
    /// command. Required on boot for newer comm boards
    /// (`slave_version >= 4.1`) before any motor telemetry will flow.
    pub async fn fdcan_reset(&mut self) -> Result<()> {
        let _span = tracing::info_span!("FDCAN reset");
        if let Err(e) = self.send_one_shot(CommandId::FdcanReset, &[0x00]).await {
            tracing::error!("Error sending FDCAN reset: {e}");
        }
        sleep(Duration::from_millis(10)).await;
        self.poll().await?;
        Ok(())
    }

    /// Set the motor-command watchdog timeout on every registered motor.
    ///
    /// The firmware will stop commanding a motor if it doesn't receive a new
    /// command within `timeout`. Pass [`Duration::ZERO`] to disable the timeout.
    pub async fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        let timeout_milliseconds = timeout_to_firmware_milliseconds(timeout)?;
        let id_max = self.pending.id_max;
        let mut payload = Vec::with_capacity(usize::from(id_max) * 2);
        for _ in 0..id_max {
            payload.extend_from_slice(&timeout_milliseconds.to_le_bytes());
        }
        self.send_one_shot(CommandId::Timeout, &payload).await
    }

    /// Read one valid response, preserving partial frames across timeouts.
    pub async fn poll(&mut self) -> Result<Option<CommandId>> {
        match timeout(DEFAULT_SERIAL_TIMEOUT, self.receive_frame()).await {
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }

    async fn receive_frame(&mut self) -> Result<CommandId> {
        let mut buffer = [0_u8; 512];
        loop {
            if let Some(frame) = self.reader.next_frame() {
                self.process_frame(&frame);
                return Ok(frame.command);
            }
            let received = self.port.read(&mut buffer).await?;
            if received == 0 {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
            }
            self.reader.extend(&buffer[..received]);
        }
    }

    /// Decode currently available bytes without waiting for another reply.
    pub async fn poll_until_empty(&mut self) -> Result<usize> {
        let mut buffer = [0_u8; 512];
        let available = self.port.bytes_to_read().map_err(io::Error::from)? as usize;
        let mut remaining = available;
        let mut frames_decoded = 0;
        loop {
            while let Some(frame) = self.reader.next_frame() {
                self.process_frame(&frame);
                frames_decoded += 1;
            }
            if remaining == 0 {
                return Ok(frames_decoded);
            }
            let chunk = remaining.min(buffer.len());
            match self.port.try_read(&mut buffer[..chunk]) {
                Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into()),
                Ok(received) => {
                    self.reader.extend(&buffer[..received]);
                    remaining -= received;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    return Ok(frames_decoded);
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    pub(crate) fn into_parts(self) -> ControllerParts {
        ControllerParts {
            port: self.port,
            motors: self.motors,
            pending: self.pending,
            reader: self.reader,
            states: self.states,
            received_at: self.received_at,
        }
    }

    /// Best-effort shutdown: stop the motors and discard any buffered bytes
    /// so the kernel can close the USB-CDC port immediately when this
    /// controller is dropped. All steps swallow their errors — this is
    /// meant to be called from a cleanup path where failing is worse than
    /// silently proceeding.
    ///
    /// Await this method before dropping a standalone controller.
    pub async fn shutdown(&mut self) {
        let _ = self.stop().await;
        let _ = self.port.clear(ClearBuffer::All);
    }

    async fn send_one_shot(&mut self, command: CommandId, payload: &[u8]) -> Result<()> {
        let _span = tracing::info_span!("Sending command to motor", command = ?command);
        self.tx_scratch.clear();
        encode_frame(command, payload, &mut self.tx_scratch)?;
        self.port.write_all(&self.tx_scratch).await?;
        Ok(())
    }

    fn process_frame(&mut self, frame: &DecodedFrame) {
        match frame.command {
            CommandId::MotorState | CommandId::MotorState2 => {
                let with_diagnostics = frame.command == CommandId::MotorState2;
                decode_motor_states(
                    &frame.payload,
                    with_diagnostics,
                    &self.motors,
                    &mut self.states,
                );
                let entry_size = if with_diagnostics {
                    MOTOR_STATE2_ENTRY_SIZE
                } else {
                    MOTOR_STATE_ENTRY_SIZE
                };
                let received_at = std::time::Instant::now();
                for entry in frame.payload.chunks_exact(entry_size) {
                    if self.motors.contains_key(&entry[0]) {
                        self.received_at.insert(entry[0], received_at);
                    }
                }
            }
            CommandId::SetNum => {
                // Reference: `p_port_version = data[2] + data[3] * 0.1`.
                if frame.payload.len() >= 4 {
                    let version =
                        f32::from(frame.payload[3]).mul_add(0.1, f32::from(frame.payload[2]));
                    self.port_version = Some(version);
                } else {
                    tracing::warn!(
                        "poll: MODE_SET_NUM reply too short ({} bytes); expected >= 4",
                        frame.payload.len(),
                    );
                }
            }
            CommandId::FunV => {
                // Reference: `p_fun_v = (fun_version)data[0]`.
                if let Some(raw) = frame.payload.first() {
                    self.fun_version = FunVersion::from_u8(*raw);
                    if self.fun_version.is_none() {
                        tracing::warn!(
                            "poll: MODE_FUN_V reply reports unknown version 0x{raw:02X}"
                        );
                    }
                } else {
                    tracing::warn!("poll: MODE_FUN_V reply had empty payload");
                }
            }
            _ => {
                tracing::warn!(
                    "poll: unsupported command byte 0x{}; dropping frame",
                    frame.command as u8
                );
                // TX-only or unsupported-for-RX commands are ignored.
            }
        }
    }
}

/// Parse `MODE_MOTOR_STATE` / `MODE_MOTOR_STATE2` payloads and apply the
/// per-motor reverse/offset adjustments.
pub(crate) fn decode_motor_states(
    payload: &[u8],
    with_mode_and_fault: bool,
    motors: &HashMap<u8, MotorConfig>,
    states: &mut HashMap<u8, MotorState>,
) {
    let entry_size = if with_mode_and_fault {
        MOTOR_STATE2_ENTRY_SIZE
    } else {
        MOTOR_STATE_ENTRY_SIZE
    };
    for entry in payload.chunks_exact(entry_size) {
        let motor_id = entry[0];
        let Some(motor) = motors.get(&motor_id) else {
            continue;
        };

        // Skip the `mode` and `fault` diagnostic bytes when the payload
        // carries them; they are not surfaced through the joint-level API.
        let rest = if with_mode_and_fault {
            &entry[3..]
        } else {
            &entry[1..]
        };

        let position_raw = i16::from_le_bytes([rest[0], rest[1]]);
        let velocity_raw = i16::from_le_bytes([rest[2], rest[3]]);
        let torque_raw = i16::from_le_bytes([rest[4], rest[5]]);

        let position = if motor.reverse {
            -position_from_i16(position_raw)
        } else {
            position_from_i16(position_raw)
        } - Angle::new::<radian>(motor.offset_rad);
        let velocity = if motor.reverse {
            -velocity_from_i16(velocity_raw)
        } else {
            velocity_from_i16(velocity_raw)
        };
        let torque = if motor.reverse {
            -torque_from_i16(torque_raw, motor.motor_type)
        } else {
            torque_from_i16(torque_raw, motor.motor_type)
        };

        states.insert(
            motor_id,
            MotorState {
                position,
                velocity,
                torque,
            },
        );
    }
}

pub(crate) struct ControllerParts {
    pub port: SerialStream,
    pub motors: HashMap<u8, MotorConfig>,
    pub pending: PendingTx,
    pub reader: FrameReader,
    pub states: HashMap<u8, MotorState>,
    pub received_at: HashMap<u8, std::time::Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::MotorType;
    use uom::si::angle::radian;
    use uom::si::angular_velocity::radian_per_second;
    use uom::si::f32::{Angle, AngularVelocity, Torque};
    use uom::si::torque::newton_meter;

    fn radians(value: f32) -> Angle {
        Angle::new::<radian>(value)
    }

    fn radians_per_second(value: f32) -> AngularVelocity {
        AngularVelocity::new::<radian_per_second>(value)
    }

    fn newton_meters(value: f32) -> Torque {
        Torque::new::<newton_meter>(value)
    }

    fn make_pending(id_max: u8) -> PendingTx {
        PendingTx::new(id_max)
    }

    #[test]
    fn staging_position_writes_sentinel_for_other_motors() {
        let mut pending = make_pending(3);
        let motor = MotorConfig::new(2, MotorType::M5047_36_2);
        pending.stage_position(motor, radians(0.0));

        assert_eq!(pending.command, Some(CommandId::Position));
        assert_eq!(pending.payload.len(), 6);
        assert_eq!(&pending.payload[0..2], &POSITION_SENTINEL_BYTES);
        assert_eq!(&pending.payload[2..4], &0i16.to_le_bytes());
        assert_eq!(&pending.payload[4..6], &POSITION_SENTINEL_BYTES);
    }

    #[test]
    fn staging_position_twice_keeps_other_slots() {
        let mut pending = make_pending(2);
        let m1 = MotorConfig::new(1, MotorType::M5047_36_2);
        let m2 = MotorConfig::new(2, MotorType::M5047_36_2);
        pending.stage_position(m1, radians(core::f32::consts::PI));
        pending.stage_position(m2, radians(0.0));

        let m1_raw = i16::from_le_bytes([pending.payload[0], pending.payload[1]]);
        let m2_raw = i16::from_le_bytes([pending.payload[2], pending.payload[3]]);
        assert_eq!(m1_raw, 5_000);
        assert_eq!(m2_raw, 0);
    }

    #[test]
    fn staging_switching_commands_resets_payload() {
        let mut pending = make_pending(2);
        let motor = MotorConfig::new(1, MotorType::M5047_36_2);
        pending.stage_position(motor, radians(0.0));
        pending.stage_pos_vel_torque_kp_kd(
            motor,
            radians(1.0),
            radians_per_second(0.5),
            newton_meters(0.0),
            10.0,
            1.0,
        );

        assert_eq!(pending.command, Some(CommandId::PosVelTorqueKpKd));
        assert_eq!(
            pending.payload.len(),
            usize::from(pending.id_max) * POS_VEL_TORQUE_KP_KD_ENTRY_SIZE
        );
        assert_eq!(
            &pending.payload[0..2],
            &position_to_i16(radians(1.0)).to_le_bytes()
        );
        assert_eq!(
            &pending.payload[2..4],
            &velocity_to_i16(radians_per_second(0.5)).to_le_bytes()
        );
    }

    #[test]
    fn staging_position_with_limits_encodes_torque_ceiling() {
        let mut pending = make_pending(1);
        let motor = MotorConfig::new(1, MotorType::M5047_36_2);

        pending.stage_pos_vel_max_torque(
            motor,
            radians(0.5),
            radians_per_second(50.0),
            newton_meters(10.0),
        );

        assert_eq!(pending.command, Some(CommandId::PosVelMaxTorque));
        assert_eq!(
            &pending.payload[0..2],
            &position_to_i16(radians(0.5)).to_le_bytes()
        );
        assert_eq!(
            &pending.payload[2..4],
            &velocity_to_i16(radians_per_second(50.0)).to_le_bytes()
        );
        assert_eq!(
            &pending.payload[4..6],
            &torque_to_i16(newton_meters(10.0), MotorType::M5047_36_2).to_le_bytes()
        );
    }

    #[test]
    fn staging_position_applies_reverse_and_offset() {
        let mut pending = make_pending(1);
        let motor = MotorConfig {
            id: 1,
            motor_type: MotorType::M5047_36_2,
            reverse: true,
            offset_rad: 0.25,
        };
        pending.stage_position(motor, radians(1.0));

        let raw = i16::from_le_bytes([pending.payload[0], pending.payload[1]]);
        let expected = position_to_i16(radians(-(1.0 + 0.25)));
        assert_eq!(raw, expected);
    }

    #[test]
    fn decoding_motor_state_restores_reverse_and_offset() {
        let mut motors = HashMap::new();
        motors.insert(
            1,
            MotorConfig {
                id: 1,
                motor_type: MotorType::M5047_36_2,
                reverse: true,
                offset_rad: 0.1,
            },
        );

        let position_raw = position_to_i16(radians(-(0.4 + 0.1)));
        let velocity_raw = velocity_to_i16(radians_per_second(-0.2));
        let torque_raw = torque_to_i16(newton_meters(-1.0), MotorType::M5047_36_2);

        let mut payload = Vec::new();
        payload.push(1u8);
        payload.extend_from_slice(&position_raw.to_le_bytes());
        payload.extend_from_slice(&velocity_raw.to_le_bytes());
        payload.extend_from_slice(&torque_raw.to_le_bytes());

        let mut states = HashMap::new();
        decode_motor_states(&payload, false, &motors, &mut states);

        let state = states.get(&1).unwrap();
        assert!((state.position.get::<radian>() - 0.4).abs() < 1e-2);
        assert!((state.velocity.get::<radian_per_second>() - 0.2).abs() < 1e-2);
        assert!((state.torque.get::<newton_meter>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn decoding_motor_state2_skips_mode_and_fault_bytes() {
        let mut motors = HashMap::new();
        motors.insert(2, MotorConfig::new(2, MotorType::M4438_30));

        let mut payload = Vec::new();
        payload.push(2u8);
        payload.push(0x05);
        payload.push(0x80);
        payload.extend_from_slice(&position_to_i16(radians(0.25)).to_le_bytes());
        payload.extend_from_slice(&velocity_to_i16(radians_per_second(0.0)).to_le_bytes());
        payload.extend_from_slice(
            &torque_to_i16(newton_meters(0.0), MotorType::M4438_30).to_le_bytes(),
        );

        let mut states = HashMap::new();
        decode_motor_states(&payload, true, &motors, &mut states);

        let state = states.get(&2).unwrap();
        assert!((state.position.get::<radian>() - 0.25).abs() < 1e-2);
    }
}

#[cfg(all(test, unix))]
mod receive_tests {
    use super::*;

    #[tokio::test]
    async fn partial_reply_survives_cancelled_poll() {
        let (mut peer, port) = SerialStream::pair().unwrap();
        let mut controller = MotorController::from_port(port, Vec::new());
        let mut frame = Vec::new();
        encode_frame(CommandId::FunV, &[FunVersion::V5 as u8], &mut frame).unwrap();
        peer.write_all(&frame[..5]).await.unwrap();
        assert!(
            timeout(Duration::from_millis(5), controller.poll())
                .await
                .is_err()
        );
        peer.write_all(&frame[5..]).await.unwrap();
        assert_eq!(controller.poll().await.unwrap(), Some(CommandId::FunV));
        assert_eq!(controller.fun_version(), Some(FunVersion::V5));
    }

    #[tokio::test]
    async fn handshake_accepts_matching_reply_after_unrelated_frame() {
        let (mut peer, port) = SerialStream::pair().unwrap();
        let mut controller = MotorController::from_port(port, Vec::new());
        let mut replies = Vec::new();
        encode_frame(CommandId::FunV, &[FunVersion::V4 as u8], &mut replies).unwrap();
        encode_frame(CommandId::SetNum, &[0, 0, 4, 6], &mut replies).unwrap();
        peer.write_all(&replies).await.unwrap();
        assert_eq!(controller.set_motor_count(1).await.unwrap(), Some(4.6));
        let mut request = [0_u8; 8];
        peer.read_exact(&mut request).await.unwrap();
        assert_eq!(request[1], CommandId::SetNum as u8);
    }
}
