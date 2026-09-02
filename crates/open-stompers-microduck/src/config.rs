//! Microduck serial configuration and default device path.

use std::time::Duration;

use serde::Deserialize;

use crate::imu::ImuMount;

/// Default Dynamixel baud rate. Matches the bus speed pinned in servo EEPROM.
pub const DEFAULT_BAUDRATE: u32 = 1_000_000;

/// Default serial path for the shared servo and IMU bus on a Radxa Zero 3W.
pub const DEFAULT_SERIAL_PATH: &str = "/dev/ttyS2";

/// Default bound for one Dynamixel transaction.
///
/// A healthy 16-device read finishes well inside this. Capping it keeps a missing
/// device from stalling the telemetry loop on the serial driver's default timeout.
pub const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_millis(30);

/// Serial path, baud rate, and bus options for Microduck hardware.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct MicroduckConfig {
    /// Dynamixel bus serial path. The fifteen servos and the IMU board share it.
    pub serial_path: String,
    /// Dynamixel bus baud rate.
    pub baudrate: u32,
    /// Background telemetry request period.
    pub motor_telemetry_period: Duration,
    /// How long to wait for a Dynamixel status packet before keeping the last sample.
    pub response_timeout: Duration,
    /// Use protocol 2.0 fast sync read (`0x8A`) for the combined IMU and servo read.
    ///
    /// Turn this off for XL330 firmware older than v46, or an IMU board that does
    /// not implement the instruction. A device that does not answer makes every
    /// combined read time out.
    pub fast_sync_read: bool,
    /// Sensor-to-trunk rotation applied to gyro and orientation.
    pub imu_mount: ImuMount,
}

impl Default for MicroduckConfig {
    fn default() -> Self {
        Self {
            serial_path: DEFAULT_SERIAL_PATH.to_owned(),
            baudrate: DEFAULT_BAUDRATE,
            motor_telemetry_period: Duration::from_millis(20),
            response_timeout: DEFAULT_RESPONSE_TIMEOUT,
            fast_sync_read: true,
            imu_mount: ImuMount::Zero3,
        }
    }
}
