//! Asynchronous serial driver for a Yesense IMU.
//!
//! Ports the read-only path of
//! `livelybot_hardware_sdk/src/yesense_ros/yesense/src/yesense_driver.cpp`:
//! open a UART, drain incoming bytes, feed them through
//! [`YesenseFrameReader`], and expose the most recent [`InertialMeasurement`].
//!
//! # Example
//!
//! ```no_run
//! use open_stompers_mini_pi::{DEFAULT_YESENSE_BAUDRATE, YesenseDriver};
//!
//! # async fn run() -> open_stompers_mini_pi::Result<()> {
//! let mut imu = YesenseDriver::open("/dev/ttyUSB0", DEFAULT_YESENSE_BAUDRATE)?;
//! loop {
//!     if imu.poll().await? > 0 {
//!         println!("inertial measurement = {:?}", imu.latest_state());
//!     }
//! }
//! # }
//! ```

use std::io;

use open_stompers_hardware::{InertialMeasurement, InertialSnapshot};
use serialport::{ClearBuffer, SerialPort};
use tokio::io::AsyncReadExt;
use tokio_serial::{SerialPortBuilderExt, SerialStream};

use crate::error::{Error, Result};
use crate::yesense_protocol::{YesenseFrameError, YesenseFrameReader, merge_frame_snapshot};

/// Default baud rate used by the Yesense IMU on the Mini Pi Plus, matching
/// the `yesense_baudrate` default in the reference ROS launch file.
pub const DEFAULT_YESENSE_BAUDRATE: u32 = 460_800;

/// Stack buffer size used by [`YesenseDriver::poll`] to drain the UART.
const READ_CHUNK_SIZE: usize = 512;

/// Read-only driver for a Yesense IMU over a serial port.
///
/// A `YesenseDriver` owns the serial port and a [`YesenseFrameReader`].
/// Call [`YesenseDriver::poll`] periodically to drain bytes from the UART
/// and update [`YesenseDriver::latest_state`].
pub struct YesenseDriver {
    port: SerialStream,
    reader: YesenseFrameReader,
    latest: InertialSnapshot,
    latest_tid: Option<u16>,
    frames_received: u64,
    checksum_failures: u64,
}

impl YesenseDriver {
    /// Open `path` at `baudrate` and return a driver ready to [`poll`] incoming data.
    pub fn open(path: &str, baudrate: u32) -> Result<Self> {
        let mut port = tokio_serial::new(path, baudrate)
            .open_native_async()
            .map_err(|source| Error::OpenPort {
                path: path.to_string(),
                source,
            })?;

        // Match the defaults the reference driver relies on (the C++ `serial`
        // library asserts DTR/RTS on open). USB-CDC bridges like the one on
        // the Mini Pi Plus gate the TX path on these signals.
        let _ = port.write_data_terminal_ready(true);
        let _ = port.write_request_to_send(true);
        let _ = port.clear(ClearBuffer::All);

        Ok(Self::from_port(port))
    }

    pub(crate) fn from_port(port: SerialStream) -> Self {
        Self {
            port,
            reader: YesenseFrameReader::new(),
            latest: InertialSnapshot {
                measurement: InertialMeasurement {
                    acceleration: [uom::si::f64::Acceleration::new::<
                        uom::si::acceleration::meter_per_second_squared,
                    >(0.0); 3],
                    angular_velocity: [uom::si::f64::AngularVelocity::new::<
                        uom::si::angular_velocity::radian_per_second,
                    >(0.0); 3],
                },
                orientation_wxyz: None,
                angular_velocity_received_at: None,
                orientation_received_at: None,
            },
            latest_tid: None,
            frames_received: 0,
            checksum_failures: 0,
        }
    }

    /// Drain whatever bytes are currently queued on the UART and merge
    /// every completed frame into the cached [`InertialMeasurement`].
    ///
    /// This is a non-blocking drain: the driver asks the OS how many bytes
    /// are already in the kernel receive buffer (matching
    /// `serial_.available()` + `serial_.read(available)` in the C++ reference
    /// driver) and only consumes that many. Bytes that arrive while `poll`
    /// is running are left for the next call, so this method always
    /// terminates even when the IMU is producing data faster than the
    /// caller's poll cadence.
    ///
    /// Returns the number of frames that were merged on this call. Checksum
    /// failures are counted separately (see
    /// [`checksum_failures`](Self::checksum_failures)) and do not surface as
    /// errors here: the reader simply resynchronises on the next valid
    /// header, matching the reference driver's warning-only behaviour.
    ///
    pub async fn poll(&mut self) -> Result<usize> {
        let _span = tracing::info_span!("yesense poll");

        let available = self.port.bytes_to_read().map_err(io::Error::from)? as usize;
        if available == 0 {
            return Ok(0);
        }

        let mut buffer = [0u8; READ_CHUNK_SIZE];
        let mut frames_this_call = 0usize;
        let mut remaining = available;

        while remaining > 0 {
            let chunk = remaining.min(buffer.len());
            match self.port.read(&mut buffer[..chunk]).await {
                Ok(0) => break,
                Ok(read) => {
                    frames_this_call += self.process_bytes(&buffer[..read]);
                    remaining = remaining.saturating_sub(read);
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) =>
                {
                    break;
                }
                Err(error) => return Err(Error::Io(error)),
            }
        }

        Ok(frames_this_call)
    }

    pub(crate) async fn receive(&mut self) -> Result<usize> {
        let mut buffer = [0_u8; READ_CHUNK_SIZE];
        let received = self.port.read(&mut buffer).await?;
        if received == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
        }
        Ok(self.process_bytes(&buffer[..received]))
    }

    fn process_bytes(&mut self, bytes: &[u8]) -> usize {
        let mut frames_this_call = 0;
        for byte in bytes {
            match self.reader.feed_byte(*byte) {
                Ok(Some(frame)) => {
                    merge_frame_snapshot(&mut self.latest, &frame.payload);
                    self.latest_tid = Some(frame.tid);
                    self.frames_received += 1;
                    frames_this_call += 1;
                }
                Ok(None) => {}
                Err(
                    YesenseFrameError::BadChecksumLow
                    | YesenseFrameError::BadChecksumHigh
                    | YesenseFrameError::EmptyPayload,
                ) => {
                    self.checksum_failures += 1;
                    tracing::warn!(
                        checksum_failures = self.checksum_failures,
                        "yesense frame rejected",
                    );
                }
            }
        }
        frames_this_call
    }

    /// Return the most recent inertial measurement.
    #[must_use]
    pub const fn latest_state(&self) -> &InertialMeasurement {
        &self.latest.measurement
    }

    /// Return the most recent inertial sample and policy-critical timestamps.
    #[must_use]
    pub const fn latest_snapshot(&self) -> &InertialSnapshot {
        &self.latest
    }

    /// Transport ID reported by the most recently merged frame, if any.
    #[must_use]
    pub const fn latest_tid(&self) -> Option<u16> {
        self.latest_tid
    }

    /// Total number of valid frames received since the driver was opened.
    #[must_use]
    pub const fn frames_received(&self) -> u64 {
        self.frames_received
    }

    /// Total number of frames rejected by the frame reader (bad checksum or
    /// zero-length payload) since the driver was opened.
    #[must_use]
    pub const fn checksum_failures(&self) -> u64 {
        self.checksum_failures
    }

    /// Serial port path that the driver was opened on, if the platform
    /// reports one.
    #[must_use]
    pub fn port_name(&self) -> Option<String> {
        self.port.name()
    }
}
