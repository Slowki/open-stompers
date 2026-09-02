//! Background serial communication and latest-value caches.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::MissedTickBehavior;

use open_stompers_hardware::{InertialMeasurement, InertialSnapshot};

use crate::controller::{ControllerParts, MotorController, PendingTx, decode_motor_states};
use crate::error::{Error, Result};
use crate::motor::{MotorConfig, MotorState};
use crate::protocol::{CommandId, FrameReader, encode_frame};
use crate::yesense_driver::YesenseDriver;

#[derive(Clone, Copy)]
pub(crate) struct MotorSnapshot {
    pub state: MotorState,
    pub received_at: Instant,
}

/// A motor state paired with the time its reply was received.
#[derive(Clone, Copy)]
pub struct ReceivedMotorState {
    /// Latest decoded motor state.
    pub state: MotorState,
    /// Host receive time for this state.
    pub received_at: Instant,
}

/// Maximum write duration, matching the motor command watchdog.
const SERIAL_WRITE_TIMEOUT: Duration = Duration::from_millis(100);

type HealthReceiver = watch::Receiver<Option<Arc<Error>>>;
type Acknowledgment = oneshot::Receiver<Result<()>>;

pub(crate) struct BoardHandle {
    pub pending: Mutex<PendingTx>,
    commands: mpsc::UnboundedSender<BoardCommand>,
    states: HashMap<u8, watch::Receiver<Option<MotorSnapshot>>>,
    health: HealthReceiver,
}

pub(crate) struct BackgroundDriver {
    pub boards: Vec<BoardHandle>,
    measurement: watch::Receiver<InertialSnapshot>,
    sensor_health: HealthReceiver,
    sensor_shutdown: watch::Sender<bool>,
    tasks: tokio::sync::Mutex<Vec<JoinHandle<()>>>,
    task_cancellations: Vec<AbortHandle>,
    closed: AtomicBool,
    shutdown_lock: tokio::sync::Mutex<()>,
}

enum BoardCommand {
    Frame {
        command: CommandId,
        payload: Vec<u8>,
        acknowledgment: oneshot::Sender<Result<()>>,
    },
    RequestStates {
        acknowledgment: oneshot::Sender<Result<()>>,
    },
    SetTelemetryEnabled {
        enabled: bool,
        acknowledgment: oneshot::Sender<Result<()>>,
    },
    Stop {
        shutdown: bool,
        acknowledgment: oneshot::Sender<Result<()>>,
    },
}

impl BoardHandle {
    #[cfg(test)]
    pub fn state_updates(&self, motor_id: u8) -> watch::Receiver<Option<MotorSnapshot>> {
        self.states[&motor_id].clone()
    }

    pub fn check_health(&self) -> Result<()> {
        check_health(&self.health)
    }

    pub fn snapshot(&self, motor_id: u8) -> Option<MotorSnapshot> {
        *self.states.get(&motor_id)?.borrow()
    }

    async fn request_states(&self) -> Result<()> {
        self.check_health()?;
        let (acknowledgment, received) = oneshot::channel();
        self.commands
            .send(BoardCommand::RequestStates { acknowledgment })
            .map_err(|_| Error::CommunicationStopped)?;
        received.await.unwrap_or(Err(Error::CommunicationStopped))
    }

    async fn set_telemetry_enabled(&self, enabled: bool) -> Result<()> {
        let (acknowledgment, received) = oneshot::channel();
        self.commands
            .send(BoardCommand::SetTelemetryEnabled {
                enabled,
                acknowledgment,
            })
            .map_err(|_| Error::CommunicationStopped)?;
        received.await.unwrap_or(Err(Error::CommunicationStopped))
    }

    pub fn flush(&self) -> Result<Option<Acknowledgment>> {
        self.check_health()?;
        let mut pending = self.pending.lock().expect("motor command lock poisoned");
        let Some((command, payload)) = pending.take_frame() else {
            return Ok(None);
        };
        let (acknowledgment, received) = oneshot::channel();
        self.commands
            .send(BoardCommand::Frame {
                command,
                payload,
                acknowledgment,
            })
            .map_err(|_| Error::CommunicationStopped)?;
        Ok(Some(received))
    }

    fn stop(&self, shutdown: bool) -> Result<Acknowledgment> {
        let mut pending = self.pending.lock().expect("motor command lock poisoned");
        pending.reset();
        let (acknowledgment, received) = oneshot::channel();
        self.commands
            .send(BoardCommand::Stop {
                shutdown,
                acknowledgment,
            })
            .map_err(|_| Error::CommunicationStopped)?;
        Ok(received)
    }
}

impl BackgroundDriver {
    #[cfg(test)]
    pub fn measurement_updates(&self) -> watch::Receiver<InertialSnapshot> {
        self.measurement.clone()
    }

    #[cfg(test)]
    pub fn tasks_finished(&self) -> bool {
        self.task_cancellations.iter().all(AbortHandle::is_finished)
    }

    #[cfg(test)]
    pub fn task_abort_handles(&self) -> Vec<tokio::task::AbortHandle> {
        self.task_cancellations.clone()
    }

    pub fn start(
        controllers: Vec<MotorController>,
        mut sensor: YesenseDriver,
        telemetry_period: Duration,
    ) -> Self {
        let mut boards = Vec::with_capacity(controllers.len());
        let mut tasks = Vec::with_capacity(controllers.len() + 1);
        for controller in controllers {
            let (board, task) = start_board(controller.into_parts(), telemetry_period);
            boards.push(board);
            tasks.push(task);
        }
        let (measurement, latest_measurement) = watch::channel(*sensor.latest_snapshot());
        let (sensor_health, latest_sensor_health) = watch::channel(None);
        let (sensor_shutdown, mut shutdown_requested) = watch::channel(false);
        tasks.push(tokio::spawn(async move {
            let result = tokio::select! {
                result = receive_measurements(&mut sensor, measurement) => result,
                _ = shutdown_requested.changed() => Ok(()),
            };
            if let Err(error) = result {
                sensor_health.send_replace(Some(Arc::new(error)));
            }
        }));
        let task_cancellations = tasks.iter().map(JoinHandle::abort_handle).collect();
        Self {
            boards,
            measurement: latest_measurement,
            sensor_health: latest_sensor_health,
            sensor_shutdown,
            tasks: tokio::sync::Mutex::new(tasks),
            task_cancellations,
            closed: AtomicBool::new(false),
            shutdown_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn check_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::CommunicationStopped);
        }
        Ok(())
    }

    pub fn read_measurement(&self) -> Result<InertialMeasurement> {
        self.check_open()?;
        check_health(&self.sensor_health)?;
        Ok(self.measurement.borrow().measurement)
    }

    pub fn read_inertial_snapshot(&self) -> Result<InertialSnapshot> {
        self.check_open()?;
        check_health(&self.sensor_health)?;
        Ok(*self.measurement.borrow())
    }

    pub async fn read_fresh_motor_states(
        &self,
        required_motors: &[(usize, u8)],
        timeout: Duration,
    ) -> Result<Vec<ReceivedMotorState>> {
        self.check_open()?;
        let started_at = Instant::now();
        let mut receivers = Vec::with_capacity(required_motors.len());
        for (board_index, motor_id) in required_motors {
            let board = self
                .boards
                .get(*board_index)
                .ok_or(Error::CommunicationStopped)?;
            let receiver = board
                .states
                .get(motor_id)
                .ok_or(Error::CommunicationStopped)?
                .clone();
            receivers.push(receiver);
        }
        let mut requests = Vec::with_capacity(self.boards.len());
        for board in &self.boards {
            requests.push(board.request_states());
        }
        tokio::time::timeout(timeout, async {
            for request in requests {
                request.await?;
            }
            let mut snapshots = Vec::with_capacity(receivers.len());
            for mut receiver in receivers {
                loop {
                    if let Some(snapshot) = *receiver.borrow()
                        && snapshot.received_at >= started_at
                    {
                        snapshots.push(ReceivedMotorState {
                            state: snapshot.state,
                            received_at: snapshot.received_at,
                        });
                        break;
                    }
                    receiver
                        .changed()
                        .await
                        .map_err(|_| Error::CommunicationStopped)?;
                }
            }
            Ok(snapshots)
        })
        .await
        .map_err(|_| {
            Error::TelemetryUnavailable(format!(
                "fresh motor telemetry timed out for motors {required_motors:?}"
            ))
        })?
    }

    pub async fn set_motor_telemetry_enabled(&self, enabled: bool) -> Result<()> {
        self.check_open()?;
        let mut updates = Vec::with_capacity(self.boards.len());
        for board in &self.boards {
            updates.push(board.set_telemetry_enabled(enabled));
        }
        for update in updates {
            update.await?;
        }
        Ok(())
    }

    pub async fn flush_all(&self) -> Result<()> {
        self.check_open()?;
        let mut acknowledgments = Vec::with_capacity(self.boards.len());
        let mut failure = None;
        for board in &self.boards {
            match board.flush() {
                Ok(Some(acknowledgment)) => acknowledgments.push(acknowledgment),
                Ok(None) => {}
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        await_acknowledgments(acknowledgments, failure).await
    }

    pub async fn stop_all(&self) -> Result<()> {
        self.check_open()?;
        self.send_stops(false).await
    }

    async fn send_stops(&self, shutdown: bool) -> Result<()> {
        let mut acknowledgments = Vec::with_capacity(self.boards.len());
        let mut failure = None;
        for board in &self.boards {
            match board.stop(shutdown) {
                Ok(acknowledgment) => acknowledgments.push(acknowledgment),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        await_acknowledgments(acknowledgments, failure).await
    }

    pub async fn shutdown(&self) -> Result<()> {
        let _guard = self.shutdown_lock.lock().await;
        let mut failure = None;
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.sensor_shutdown.send_replace(true);
            failure = self.send_stops(true).await.err();
        }
        let mut tasks = self.tasks.lock().await;
        while let Some(task) = tasks.last_mut() {
            if let Err(error) = task.await {
                failure.get_or_insert(Error::TaskFailed(error));
            }
            tasks.pop();
        }
        for board in &self.boards {
            if let Some(error) = board.health.borrow().as_ref() {
                failure.get_or_insert_with(|| Error::CommunicationFailed(Arc::clone(error)));
            }
        }
        if let Some(error) = self.sensor_health.borrow().as_ref() {
            failure.get_or_insert_with(|| Error::CommunicationFailed(Arc::clone(error)));
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for BackgroundDriver {
    fn drop(&mut self) {
        for task in &self.task_cancellations {
            task.abort();
        }
    }
}

fn check_health(health: &HealthReceiver) -> Result<()> {
    if let Some(error) = health.borrow().as_ref() {
        return Err(Error::CommunicationFailed(Arc::clone(error)));
    }
    if health.has_changed().is_err() {
        return Err(Error::CommunicationStopped);
    }
    Ok(())
}

async fn await_acknowledgments(
    acknowledgments: Vec<Acknowledgment>,
    mut failure: Option<Error>,
) -> Result<()> {
    for acknowledgment in acknowledgments {
        if let Err(error) = acknowledgment
            .await
            .unwrap_or(Err(Error::CommunicationStopped))
        {
            failure.get_or_insert(error);
        }
    }
    failure.map_or(Ok(()), Err)
}

fn start_board(
    parts: ControllerParts,
    telemetry_period: Duration,
) -> (BoardHandle, JoinHandle<()>) {
    let ControllerParts {
        port,
        motors,
        pending,
        reader,
        states,
        received_at,
    } = parts;
    let mut latest_states = HashMap::with_capacity(motors.len());
    let mut updates = HashMap::with_capacity(motors.len());
    for motor_id in motors.keys() {
        let initial = states.get(motor_id).map(|state| MotorSnapshot {
            state: *state,
            received_at: received_at[motor_id],
        });
        let (sender, receiver) = watch::channel(initial);
        latest_states.insert(*motor_id, receiver);
        updates.insert(*motor_id, sender);
    }
    let (commands, requests) = mpsc::unbounded_channel();
    let (health, latest_health) = watch::channel(None);
    let (read_port, write_port) = tokio::io::split(port);
    let task = tokio::spawn(async move {
        let result = tokio::select! {
            result = receive_motor_states(read_port, reader, motors, updates) => result,
            result = write_motor_commands(write_port, requests, telemetry_period) => result,
        };
        if let Err(error) = result {
            health.send_replace(Some(Arc::new(error)));
        }
    });
    (
        BoardHandle {
            pending: Mutex::new(pending),
            commands,
            states: latest_states,
            health: latest_health,
        },
        task,
    )
}

async fn receive_motor_states(
    mut port: impl AsyncRead + Unpin,
    mut reader: FrameReader,
    motors: HashMap<u8, MotorConfig>,
    updates: HashMap<u8, watch::Sender<Option<MotorSnapshot>>>,
) -> Result<()> {
    let mut buffer = [0_u8; 512];
    let mut states = HashMap::with_capacity(motors.len());
    loop {
        while let Some(frame) = reader.next_frame() {
            if !matches!(
                frame.command,
                CommandId::MotorState | CommandId::MotorState2
            ) {
                continue;
            }
            states.clear();
            decode_motor_states(
                &frame.payload,
                frame.command == CommandId::MotorState2,
                &motors,
                &mut states,
            );
            let received_at = Instant::now();
            for (motor_id, state) in &states {
                updates[motor_id].send_replace(Some(MotorSnapshot {
                    state: *state,
                    received_at,
                }));
            }
        }
        let received = port.read(&mut buffer).await?;
        if received == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
        }
        reader.extend(&buffer[..received]);
    }
}

async fn write_motor_commands(
    mut port: impl AsyncWrite + Unpin,
    mut commands: mpsc::UnboundedReceiver<BoardCommand>,
    telemetry_period: Duration,
) -> Result<()> {
    let mut ticker = tokio::time::interval(telemetry_period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut buffer = Vec::with_capacity(256);
    let mut telemetry_enabled = true;
    loop {
        tokio::select! {
            request = commands.recv() => {
                let Some(request) = request else { return Ok(()); };
                let (command, payload, acknowledgment, shutdown) = match request {
                    BoardCommand::Frame { command, payload, acknowledgment } =>
                        (command, payload, acknowledgment, false),
                    BoardCommand::Stop { shutdown, acknowledgment } =>
                        (CommandId::Stop, vec![0x7F], acknowledgment, shutdown),
                    BoardCommand::RequestStates { acknowledgment } => {
                        let result = write_frame(&mut port, CommandId::MotorState2, &[0x7F], &mut buffer).await;
                        let _ = acknowledgment.send(result);
                        continue;
                    }
                    BoardCommand::SetTelemetryEnabled { enabled, acknowledgment } => {
                        telemetry_enabled = enabled;
                        let _ = acknowledgment.send(Ok(()));
                        continue;
                    }
                };
                if let Err(error) = write_frame(&mut port, command, &payload, &mut buffer).await {
                    let error = Arc::new(error);
                    let _ = acknowledgment.send(Err(Error::CommunicationFailed(Arc::clone(&error))));
                    return Err(Error::CommunicationFailed(error));
                }
                let _ = acknowledgment.send(Ok(()));
                if shutdown { return Ok(()); }
            }
            _ = ticker.tick(), if telemetry_enabled => {
                write_frame(&mut port, CommandId::MotorState2, &[0x7F], &mut buffer).await?;
            }
        }
    }
}

async fn write_frame(
    port: &mut (impl AsyncWrite + Unpin),
    command: CommandId,
    payload: &[u8],
    buffer: &mut Vec<u8>,
) -> Result<()> {
    buffer.clear();
    encode_frame(command, payload, buffer)?;
    tokio::time::timeout(SERIAL_WRITE_TIMEOUT, port.write_all(buffer))
        .await
        .map_err(|_| io::Error::from(io::ErrorKind::TimedOut))??;
    Ok(())
}

async fn receive_measurements(
    sensor: &mut YesenseDriver,
    updates: watch::Sender<InertialSnapshot>,
) -> Result<()> {
    loop {
        if sensor.receive().await? > 0 {
            updates.send_replace(*sensor.latest_snapshot());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    use super::*;

    #[tokio::test(start_paused = true)]
    async fn requests_follow_configured_period_and_skip_missed_ticks() {
        let (mut peer, port) = tokio::io::duplex(1024);
        let (_commands, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(write_motor_commands(
            port,
            requests,
            Duration::from_millis(20),
        ));
        let mut request = Vec::new();
        encode_frame(CommandId::MotorState2, &[0x7F], &mut request).unwrap();
        let mut received = vec![0_u8; request.len()];
        tokio::task::yield_now().await;
        peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, request);
        tokio::time::advance(Duration::from_millis(19)).await;
        assert_read_pending(&mut peer);
        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, request);
        tokio::time::advance(Duration::from_millis(100)).await;
        tokio::task::yield_now().await;
        peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, request);
        assert_read_pending(&mut peer);
        tokio::time::advance(Duration::from_millis(20)).await;
        tokio::task::yield_now().await;
        peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, request);
        task.abort();
    }

    fn assert_read_pending(peer: &mut tokio::io::DuplexStream) {
        let mut byte = [0_u8; 1];
        let mut future = pin!(peer.read(&mut byte));
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    }

    #[tokio::test]
    async fn command_acknowledgments_wait_for_complete_writes_and_preserve_order() {
        let (mut peer, port) = tokio::io::duplex(1);
        let (commands, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(write_motor_commands(
            port,
            requests,
            Duration::from_secs(60),
        ));
        let mut initial_request = [0_u8; 8];
        peer.read_exact(&mut initial_request).await.unwrap();
        let (acknowledgment, mut received_acknowledgment) = oneshot::channel();
        commands
            .send(BoardCommand::Frame {
                command: CommandId::Position,
                payload: vec![1, 2, 3, 4],
                acknowledgment,
            })
            .unwrap();
        let (stop_acknowledgment, stop_received) = oneshot::channel();
        commands
            .send(BoardCommand::Stop {
                shutdown: true,
                acknowledgment: stop_acknowledgment,
            })
            .unwrap();
        tokio::task::yield_now().await;
        assert!(matches!(
            received_acknowledgment.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let mut expected = Vec::new();
        encode_frame(CommandId::Position, &[1, 2, 3, 4], &mut expected).unwrap();
        let mut received = vec![0_u8; expected.len()];
        peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, expected);
        received_acknowledgment.await.unwrap().unwrap();
        expected.clear();
        encode_frame(CommandId::Stop, &[0x7F], &mut expected).unwrap();
        received.resize(expected.len(), 0);
        peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, expected);
        stop_received.await.unwrap().unwrap();
        task.await.unwrap().unwrap();
    }
    #[tokio::test(start_paused = true)]
    async fn replies_publish_while_command_writes_are_blocked() {
        let (mut peer, port) = tokio::io::duplex(128);
        let (read_port, write_port) = tokio::io::split(port);
        let motor = MotorConfig::new(1, crate::motor::MotorType::M3536_32);
        let (updates, mut latest) = watch::channel(None);
        let (commands, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            tokio::select! {
                result = receive_motor_states(
                    read_port,
                    FrameReader::new(),
                    HashMap::from([(1, motor)]),
                    HashMap::from([(1, updates)]),
                ) => result,
                result = write_motor_commands(write_port, requests, Duration::from_secs(60)) => result,
            }
        });
        let mut initial_request = [0_u8; 8];
        peer.read_exact(&mut initial_request).await.unwrap();
        let (acknowledgment, mut acknowledged) = oneshot::channel();
        commands
            .send(BoardCommand::Frame {
                command: CommandId::Position,
                payload: vec![0; 256],
                acknowledgment,
            })
            .unwrap();
        tokio::task::yield_now().await;
        assert!(matches!(
            acknowledged.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let mut frame = Vec::new();
        encode_frame(
            CommandId::MotorState2,
            &[1, 0, 0, 1, 0, 2, 0, 3, 0],
            &mut frame,
        )
        .unwrap();
        peer.write_all(&frame).await.unwrap();
        latest.changed().await.unwrap();
        assert!(latest.borrow().is_some());
        assert!(matches!(
            acknowledged.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        tokio::time::advance(SERIAL_WRITE_TIMEOUT).await;
        assert!(task.await.unwrap().is_err());
        assert!(acknowledged.await.unwrap().is_err());
    }
}
