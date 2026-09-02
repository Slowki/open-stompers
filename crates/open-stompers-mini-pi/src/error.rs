//! Error type for the Mini Pi platform crate.

use std::time::Duration;

use crate::protocol::FrameError;
use crate::yesense_protocol::YesenseFrameError;
use thiserror::Error;

/// Result alias used by public APIs in this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Hardware failures surfaced by the Mini Pi Plus driver.
#[derive(Debug, Error)]
pub enum Error {
    /// Failed to open or configure the underlying serial port.
    #[error("failed to open serial port {path:?}: {source}")]
    OpenPort {
        /// Path that was attempted.
        path: String,
        /// Underlying `serialport` error.
        #[source]
        source: serialport::Error,
    },

    /// A telemetry request period must be positive.
    #[error("motor telemetry request period must be greater than zero")]
    InvalidTelemetryPeriod,

    /// No reply has been received for this motor.
    #[error("telemetry is unavailable for joint {0:?}")]
    TelemetryUnavailable(String),

    /// Communication has been shut down or its task terminated.
    #[error("background communication task stopped")]
    CommunicationStopped,

    /// A background communication task failed.
    #[error("background communication failed: {0}")]
    CommunicationFailed(#[source] std::sync::Arc<Error>),

    /// A communication task panicked or was cancelled.
    #[error("communication task failed: {0}")]
    TaskFailed(#[from] tokio::task::JoinError),

    /// Serial port I/O failure during read or write.
    #[error("serial I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The caller referenced a motor id that was not registered at construction.
    #[error("unknown motor id {0}")]
    UnknownMotorId(u8),

    /// The caller referenced a joint name that is not part of the Mini Pi Plus topology.
    #[error("unknown joint name {0:?}")]
    UnknownJoint(String),

    /// A motor command contains a non-finite or out-of-range value.
    #[error("invalid motor command value for joint {0:?}")]
    InvalidCommandValue(String),

    /// The motor-command watchdog duration cannot be encoded as firmware milliseconds.
    #[error("motor command timeout {timeout:?} cannot be represented as i16 milliseconds")]
    TimeoutOutOfRange {
        /// Requested watchdog timeout.
        timeout: Duration,
    },

    /// The caller tried to mix incompatible command types without flushing.
    #[error("pending command {pending:?} is incompatible with new command {requested:?}")]
    CommandTypeMismatch {
        /// Command id already staged in the TX buffer.
        pending: crate::protocol::CommandId,
        /// Command id the caller is now trying to stage.
        requested: crate::protocol::CommandId,
    },

    /// Frame encode failure (for example, oversize payload).
    #[error("frame encode error: {0:?}")]
    Frame(FrameError),

    /// Yesense IMU frame decode failure (bad checksum or empty payload).
    #[error("yesense frame error: {0:?}")]
    YesenseFrame(YesenseFrameError),
}

impl From<FrameError> for Error {
    fn from(value: FrameError) -> Self {
        Self::Frame(value)
    }
}

impl From<YesenseFrameError> for Error {
    fn from(value: YesenseFrameError) -> Self {
        Self::YesenseFrame(value)
    }
}
