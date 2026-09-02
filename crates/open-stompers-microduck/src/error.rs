//! Error type for the Microduck platform crate.

use std::sync::Arc;

use thiserror::Error;

use crate::protocol::ProtocolError;

/// Result alias used by public APIs in this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Hardware failures surfaced by the Microduck driver.
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

    /// A Dynamixel response timeout must be positive.
    #[error("dynamixel response timeout must be greater than zero")]
    InvalidResponseTimeout,

    /// No reply has been received for this motor.
    #[error("telemetry is unavailable for joint {0:?}")]
    TelemetryUnavailable(String),

    /// Communication has been shut down or its task terminated.
    #[error("background communication task stopped")]
    CommunicationStopped,

    /// A background communication task failed.
    #[error("background communication failed: {0}")]
    CommunicationFailed(#[source] Arc<Error>),

    /// A communication task panicked or was cancelled.
    #[error("communication task failed: {0}")]
    TaskFailed(#[from] tokio::task::JoinError),

    /// Serial port I/O failure during read or write.
    #[error("serial I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The caller referenced a joint name that is not part of the Microduck topology.
    #[error("unknown joint name {0:?}")]
    UnknownJoint(String),

    /// A motor command contains a non-finite or out-of-range value.
    #[error("invalid motor command value for joint {0:?}")]
    InvalidCommandValue(String),

    /// A Dynamixel packet could not be decoded.
    #[error("dynamixel protocol error: {0}")]
    Protocol(#[from] ProtocolError),
}
