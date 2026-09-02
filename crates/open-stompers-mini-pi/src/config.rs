//! Mini Pi Plus serial configuration and default device paths.

/// Number of Mini Pi Plus motor serial ports.
pub const SERIAL_PORT_COUNT: usize = 6;
/// Default motor communication-board baud rate.
pub const DEFAULT_MOTOR_BAUDRATE: u32 = 4_000_000;
/// Default serial path for each motor communication board.
pub const DEFAULT_MOTOR_SERIAL_PATHS: [&str; SERIAL_PORT_COUNT] = [
    "/dev/ttyACM0",
    "/dev/ttyACM1",
    "/dev/ttyACM2",
    "/dev/ttyACM3",
    "/dev/ttyACM4",
    "/dev/ttyACM5",
];
/// Default serial path for the Yesense IMU.
pub const DEFAULT_IMU_SERIAL_PATH: &str = "/dev/ttyUSB0";

/// Serial paths and baud rates for Mini Pi Plus hardware.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct MiniPiPlusConfig {
    /// Motor communication-board serial paths in CAN port order.
    pub motor_serial_paths: [String; SERIAL_PORT_COUNT],
    /// Yesense IMU serial path.
    pub imu_serial_path: String,
    /// Motor communication-board baud rate.
    pub motor_baudrate: u32,
    /// Background motor telemetry request period in milliseconds.
    pub motor_telemetry_period: std::time::Duration,
    /// Yesense IMU baud rate.
    pub imu_baudrate: u32,
}

impl Default for MiniPiPlusConfig {
    fn default() -> Self {
        Self {
            motor_serial_paths: DEFAULT_MOTOR_SERIAL_PATHS.map(str::to_owned),
            imu_serial_path: DEFAULT_IMU_SERIAL_PATH.to_owned(),
            motor_baudrate: DEFAULT_MOTOR_BAUDRATE,
            motor_telemetry_period: std::time::Duration::from_millis(20),
            imu_baudrate: crate::yesense_driver::DEFAULT_YESENSE_BAUDRATE,
        }
    }
}
