//! Background Dynamixel transactions and latest-value caches.
//!
//! One task owns the bus. It periodically sync-reads the IMU board and every servo,
//! and applies staged goal positions when asked. Trait reads return the cache
//! immediately.

use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use open_stompers_hardware::InertialMeasurement;
use serialport::{ClearBuffer, SerialPort};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::MissedTickBehavior;
use tokio_serial::SerialStream;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{Acceleration, AngularVelocity};

use crate::error::{Error, Result};
use crate::imu::{DecodedImu, IMU_BLOCK_LEN, ImuDecoder, ImuMount};
use crate::model::{self, IMU_ID, JOINT_DEFINITIONS};
use crate::protocol::{
    BROADCAST_ID, GOAL_POSITION_ADDR, GOAL_POSITION_LEN, INSTRUCTION_FAST_SYNC_READ,
    INSTRUCTION_SYNC_READ, INSTRUCTION_SYNC_WRITE, POSITION_D_GAIN_ADDR, POSITION_GAIN_LEN,
    PRESENT_BLOCK_ADDR, PRESENT_BLOCK_LEN, PacketReader, RawPacket, TORQUE_ENABLE_ADDR,
    encode_instruction, parse_fast_sync_read, status_body, sync_read_params, sync_write_params,
};

const SERIAL_WRITE_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug)]
pub(crate) struct MotorSnapshot {
    pub position_rad: f64,
    pub velocity_rad_per_second: f64,
    pub torque_newton_meters: f64,
    pub received_at: Instant,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImuSnapshot {
    pub measurement: InertialMeasurement,
    pub orientation_wxyz: Option<[f64; 4]>,
    pub received_at: Option<Instant>,
}

struct Pending {
    positions: HashMap<u8, i32>,
    gains: HashMap<u8, (u16, u16)>,
}

enum BusCommand {
    Apply {
        positions: Vec<(u8, i32)>,
        gains: Vec<(u8, u16, u16)>,
        acknowledgment: oneshot::Sender<Result<()>>,
    },
    Stop {
        shutdown: bool,
        acknowledgment: oneshot::Sender<Result<()>>,
    },
}

/// Settings for the background bus task.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BusSettings {
    /// Period between telemetry reads.
    pub telemetry_period: Duration,
    /// Bound for one status-packet read.
    pub response_timeout: Duration,
    /// Use instruction `0x8A` instead of a per-device sync read.
    pub fast_sync_read: bool,
    /// IMU board mounting rotation.
    pub mount: ImuMount,
}

pub(crate) struct BackgroundDriver {
    pending: Mutex<Pending>,
    commands: mpsc::UnboundedSender<BusCommand>,
    motors: HashMap<u8, watch::Receiver<Option<MotorSnapshot>>>,
    imu: watch::Receiver<ImuSnapshot>,
    health: watch::Receiver<Option<Arc<Error>>>,
    tasks: tokio::sync::Mutex<Vec<JoinHandle<()>>>,
    task_cancellations: Vec<AbortHandle>,
    closed: AtomicBool,
    shutdown_lock: tokio::sync::Mutex<()>,
}

impl BackgroundDriver {
    pub(crate) fn start(port: SerialStream, settings: BusSettings) -> Self {
        let mut motors = HashMap::with_capacity(JOINT_DEFINITIONS.len());
        let mut updates = HashMap::with_capacity(JOINT_DEFINITIONS.len());
        for joint in JOINT_DEFINITIONS {
            let (sender, receiver) = watch::channel(None);
            motors.insert(joint.id, receiver);
            updates.insert(joint.id, sender);
        }
        let (imu_sender, imu) = watch::channel(ImuSnapshot {
            measurement: zero_measurement(),
            orientation_wxyz: None,
            received_at: None,
        });
        let (health_sender, health) = watch::channel(None);
        let (commands, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let result = run_bus(port, requests, updates, imu_sender, settings).await;
            if let Err(error) = result {
                health_sender.send_replace(Some(Arc::new(error)));
            }
        });
        let task_cancellations = vec![task.abort_handle()];
        Self {
            pending: Mutex::new(Pending {
                positions: HashMap::new(),
                gains: HashMap::new(),
            }),
            commands,
            motors,
            imu,
            health,
            tasks: tokio::sync::Mutex::new(vec![task]),
            task_cancellations,
            closed: AtomicBool::new(false),
            shutdown_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub(crate) fn check_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::CommunicationStopped);
        }
        self.check_health()
    }

    pub(crate) fn snapshot(&self, motor_id: u8) -> Option<MotorSnapshot> {
        self.motors.get(&motor_id).and_then(|state| *state.borrow())
    }

    #[cfg(test)]
    pub(crate) fn state_updates(&self, motor_id: u8) -> watch::Receiver<Option<MotorSnapshot>> {
        self.motors[&motor_id].clone()
    }

    pub(crate) fn imu_snapshot(&self) -> ImuSnapshot {
        *self.imu.borrow()
    }

    pub(crate) fn stage(&self, motor_id: u8, position: i32, proportional: u16, derivative: u16) {
        let mut pending = self.pending.lock().expect("motor command lock poisoned");
        pending.positions.insert(motor_id, position);
        pending.gains.insert(motor_id, (proportional, derivative));
    }

    pub(crate) async fn flush(&self) -> Result<()> {
        self.check_open()?;
        let (positions, gains) = {
            let mut pending = self.pending.lock().expect("motor command lock poisoned");
            let positions = pending.positions.drain().collect::<Vec<_>>();
            let gains = pending
                .gains
                .drain()
                .map(|(id, (proportional, derivative))| (id, proportional, derivative))
                .collect::<Vec<_>>();
            (positions, gains)
        };
        if positions.is_empty() {
            return Ok(());
        }
        let (acknowledgment, received) = oneshot::channel();
        self.commands
            .send(BusCommand::Apply {
                positions,
                gains,
                acknowledgment,
            })
            .map_err(|_| Error::CommunicationStopped)?;
        received.await.unwrap_or(Err(Error::CommunicationStopped))
    }

    pub(crate) async fn stop_all(&self) -> Result<()> {
        self.check_open()?;
        self.send_stop(false).await
    }

    async fn send_stop(&self, shutdown: bool) -> Result<()> {
        self.pending
            .lock()
            .expect("motor command lock poisoned")
            .positions
            .clear();
        self.pending
            .lock()
            .expect("motor command lock poisoned")
            .gains
            .clear();
        let (acknowledgment, received) = oneshot::channel();
        self.commands
            .send(BusCommand::Stop {
                shutdown,
                acknowledgment,
            })
            .map_err(|_| Error::CommunicationStopped)?;
        received.await.unwrap_or(Err(Error::CommunicationStopped))
    }

    pub(crate) async fn shutdown(&self) -> Result<()> {
        let _guard = self.shutdown_lock.lock().await;
        let mut failure = None;
        if !self.closed.swap(true, Ordering::AcqRel) {
            failure = self.send_stop(true).await.err();
        }
        let mut tasks = self.tasks.lock().await;
        while let Some(task) = tasks.last_mut() {
            if let Err(error) = task.await {
                failure.get_or_insert(Error::TaskFailed(error));
            }
            tasks.pop();
        }
        if let Some(error) = self.health.borrow().as_ref() {
            failure.get_or_insert_with(|| Error::CommunicationFailed(Arc::clone(error)));
        }
        failure.map_or(Ok(()), Err)
    }

    fn check_health(&self) -> Result<()> {
        if let Some(error) = self.health.borrow().as_ref() {
            return Err(Error::CommunicationFailed(Arc::clone(error)));
        }
        if self.health.has_changed().is_err() {
            return Err(Error::CommunicationStopped);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn tasks_finished(&self) -> bool {
        self.task_cancellations.iter().all(AbortHandle::is_finished)
    }

    #[cfg(test)]
    pub(crate) fn task_abort_handles(&self) -> Vec<AbortHandle> {
        self.task_cancellations.clone()
    }
}

impl Drop for BackgroundDriver {
    fn drop(&mut self) {
        for task in &self.task_cancellations {
            task.abort();
        }
    }
}

async fn run_bus(
    mut port: SerialStream,
    mut commands: mpsc::UnboundedReceiver<BusCommand>,
    motors: HashMap<u8, watch::Sender<Option<MotorSnapshot>>>,
    imu: watch::Sender<ImuSnapshot>,
    settings: BusSettings,
) -> Result<()> {
    let mut ticker = tokio::time::interval(settings.telemetry_period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut reader = PacketReader::new();
    let mut decoder = ImuDecoder::new(settings.mount);
    let mut enabled = HashSet::new();
    let mut applied_gains: HashMap<u8, (u16, u16)> = HashMap::new();
    let ids = model::telemetry_ids();
    let mut missed_reads = 0_u64;
    loop {
        tokio::select! {
            request = commands.recv() => {
                let Some(request) = request else { return Ok(()) };
                match request {
                    BusCommand::Apply { positions, gains, acknowledgment } => {
                        if let Err(error) = apply_targets(
                            &mut port,
                            &positions,
                            &gains,
                            &mut enabled,
                            &mut applied_gains,
                        ).await {
                            let _ = acknowledgment.send(Err(clone_io(&error)));
                            return Err(error);
                        }
                        let _ = acknowledgment.send(Ok(()));
                    }
                    BusCommand::Stop { shutdown, acknowledgment } => {
                        enabled.clear();
                        applied_gains.clear();
                        if let Err(error) = write_torque(&mut port, false).await {
                            let _ = acknowledgment.send(Err(clone_io(&error)));
                            return Err(error);
                        }
                        let _ = acknowledgment.send(Ok(()));
                        if shutdown {
                            return Ok(());
                        }
                    }
                }
            }
            _ = ticker.tick() => {
                match read_telemetry(
                    &mut port,
                    &mut reader,
                    &ids,
                    settings.fast_sync_read,
                    settings.response_timeout,
                ).await {
                    Ok(blocks) => {
                        missed_reads = 0;
                        publish_blocks(&blocks, &ids, &motors, &imu, &mut decoder);
                    }
                    Err(error) if is_disconnect(&error) => return Err(error),
                    Err(error) => {
                        missed_reads = missed_reads.saturating_add(1);
                        if missed_reads == 1 || missed_reads.is_multiple_of(50) {
                            tracing::warn!(
                                consecutive = missed_reads,
                                %error,
                                "dynamixel telemetry read failed; keeping the last sample"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn publish_blocks(
    blocks: &[Vec<u8>],
    ids: &[u8],
    motors: &HashMap<u8, watch::Sender<Option<MotorSnapshot>>>,
    imu: &watch::Sender<ImuSnapshot>,
    decoder: &mut ImuDecoder,
) {
    let received_at = Instant::now();
    for (id, block) in ids.iter().zip(blocks) {
        if *id == IMU_ID && block.len() == IMU_BLOCK_LEN {
            let mut raw = [0_u8; IMU_BLOCK_LEN];
            raw.copy_from_slice(block);
            let DecodedImu {
                measurement,
                orientation_wxyz,
            } = decoder.decode(&raw);
            imu.send_replace(ImuSnapshot {
                measurement,
                orientation_wxyz,
                received_at: Some(received_at),
            });
            continue;
        }
        if block.len() != usize::from(PRESENT_BLOCK_LEN) {
            continue;
        }
        let Some(update) = motors.get(id) else {
            continue;
        };
        let current = i16::from_le_bytes([block[2], block[3]]);
        let velocity = i32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let position = i32::from_le_bytes([block[8], block[9], block[10], block[11]]);
        update.send_replace(Some(MotorSnapshot {
            position_rad: model::position_from_raw(position).get::<uom::si::angle::radian>(),
            velocity_rad_per_second: model::velocity_from_raw(velocity)
                .get::<uom::si::angular_velocity::radian_per_second>(),
            torque_newton_meters: model::torque_from_raw(current)
                .get::<uom::si::torque::newton_meter>(),
            received_at,
        }));
    }
}

async fn read_telemetry(
    port: &mut SerialStream,
    reader: &mut PacketReader,
    ids: &[u8],
    fast_sync_read: bool,
    response_timeout: Duration,
) -> Result<Vec<Vec<u8>>> {
    let instruction = if fast_sync_read {
        INSTRUCTION_FAST_SYNC_READ
    } else {
        INSTRUCTION_SYNC_READ
    };
    let request = encode_instruction(
        BROADCAST_ID,
        instruction,
        &sync_read_params(PRESENT_BLOCK_ADDR, PRESENT_BLOCK_LEN, ids),
    );
    write_bytes(port, &request).await?;
    if fast_sync_read {
        let packet = read_packet(port, reader, response_timeout).await?;
        return parse_fast_sync_read(&packet, ids, PRESENT_BLOCK_LEN).map_err(Error::from);
    }
    let mut blocks = Vec::with_capacity(ids.len());
    for id in ids {
        let packet = read_packet(port, reader, response_timeout).await?;
        if packet.id != *id {
            return Err(Error::Protocol(
                crate::protocol::ProtocolError::UnexpectedId {
                    expected: *id,
                    actual: packet.id,
                },
            ));
        }
        let (_error, params) = status_body(&packet)?;
        blocks.push(params);
    }
    Ok(blocks)
}

async fn apply_targets(
    port: &mut SerialStream,
    positions: &[(u8, i32)],
    gains: &[(u8, u16, u16)],
    enabled: &mut HashSet<u8>,
    applied_gains: &mut HashMap<u8, (u16, u16)>,
) -> Result<()> {
    let changed_gains = gains
        .iter()
        .copied()
        .filter(|(id, proportional, derivative)| {
            applied_gains.get(id) != Some(&(*proportional, *derivative))
        })
        .collect::<Vec<_>>();
    if !changed_gains.is_empty() {
        let encoded = changed_gains
            .iter()
            .map(|(id, proportional, derivative)| {
                let mut data = [0_u8; 6];
                data[..2].copy_from_slice(&derivative.to_le_bytes());
                data[4..6].copy_from_slice(&proportional.to_le_bytes());
                debug_assert_eq!(data.len(), usize::from(POSITION_GAIN_LEN));
                (*id, data)
            })
            .collect::<Vec<_>>();
        let values = encoded
            .iter()
            .map(|(id, data)| (*id, data.as_slice()))
            .collect::<Vec<_>>();
        write_sync(port, POSITION_D_GAIN_ADDR, &values).await?;
        for (id, proportional, derivative) in changed_gains {
            applied_gains.insert(id, (proportional, derivative));
        }
    }
    let enable = positions
        .iter()
        .filter(|(id, _)| !enabled.contains(id))
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    if !enable.is_empty() {
        let enabled_byte = [1_u8];
        let values = enable
            .iter()
            .map(|id| (*id, enabled_byte.as_slice()))
            .collect::<Vec<_>>();
        write_sync(port, TORQUE_ENABLE_ADDR, &values).await?;
        enabled.extend(enable);
    }
    let encoded = positions
        .iter()
        .map(|(id, position)| (*id, position.to_le_bytes()))
        .collect::<Vec<_>>();
    let values = encoded
        .iter()
        .map(|(id, data)| (*id, data.as_slice()))
        .collect::<Vec<_>>();
    debug_assert!(
        values
            .iter()
            .all(|(_, data)| data.len() == usize::from(GOAL_POSITION_LEN))
    );
    write_sync(port, GOAL_POSITION_ADDR, &values).await
}

async fn write_torque(port: &mut SerialStream, enable: bool) -> Result<()> {
    let value = [u8::from(enable)];
    let values = JOINT_DEFINITIONS
        .iter()
        .map(|joint| (joint.id, value.as_slice()))
        .collect::<Vec<_>>();
    write_sync(port, TORQUE_ENABLE_ADDR, &values).await
}

async fn write_sync(port: &mut SerialStream, address: u16, values: &[(u8, &[u8])]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let packet = encode_instruction(
        BROADCAST_ID,
        INSTRUCTION_SYNC_WRITE,
        &sync_write_params(address, values),
    );
    write_bytes(port, &packet).await
}

async fn write_bytes(port: &mut SerialStream, bytes: &[u8]) -> Result<()> {
    tokio::time::timeout(SERIAL_WRITE_TIMEOUT, port.write_all(bytes))
        .await
        .map_err(|_| io::Error::from(io::ErrorKind::TimedOut))??;
    Ok(())
}

async fn read_packet(
    port: &mut SerialStream,
    reader: &mut PacketReader,
    timeout: Duration,
) -> Result<RawPacket> {
    let deadline = Instant::now() + timeout;
    let mut buffer = [0_u8; 256];
    loop {
        if let Some(packet) = reader.next_packet() {
            return Ok(packet);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            discard_input(port, reader);
            return Err(io::Error::from(io::ErrorKind::TimedOut).into());
        }
        match tokio::time::timeout(remaining, port.read(&mut buffer)).await {
            Err(_) => {
                discard_input(port, reader);
                return Err(io::Error::from(io::ErrorKind::TimedOut).into());
            }
            Ok(Err(error)) => return Err(error.into()),
            Ok(Ok(0)) => {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
            }
            Ok(Ok(received)) => reader.push(&buffer[..received]),
        }
    }
}

fn discard_input(port: &SerialStream, reader: &mut PacketReader) {
    reader.clear();
    let _ = port.clear(ClearBuffer::Input);
}

fn clone_io(error: &Error) -> Error {
    match error {
        Error::Io(error) => Error::Io(io::Error::new(error.kind(), error.to_string())),
        other => Error::Io(io::Error::other(other.to_string())),
    }
}

fn is_disconnect(error: &Error) -> bool {
    match error {
        Error::Io(error) => !matches!(
            error.kind(),
            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
        ),
        Error::Protocol(_) => false,
        _ => true,
    }
}

fn zero_measurement() -> InertialMeasurement {
    InertialMeasurement {
        acceleration: [Acceleration::new::<meter_per_second_squared>(0.0); 3],
        angular_velocity: [AngularVelocity::new::<radian_per_second>(0.0); 3],
    }
}
