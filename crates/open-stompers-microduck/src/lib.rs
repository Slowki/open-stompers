//! Platform-specific Microduck motor and IMU drivers.
//!
//! The fifteen XL330 servos and the `imu_to_dxl` board share one Dynamixel bus.
//! [`Microduck`] caches that bus in the background and exposes the same motor and
//! inertial-sensor traits as the Mini Pi Plus driver.

mod background;
pub mod config;
mod error;
mod imu;
mod microduck;
mod model;
mod protocol;

pub mod transforms;

pub use config::{
    DEFAULT_BAUDRATE, DEFAULT_RESPONSE_TIMEOUT, DEFAULT_SERIAL_PATH, MicroduckConfig,
};
pub use error::{Error, Result};
pub use imu::ImuMount;
pub use microduck::{Microduck, MicroduckMotor};
pub use model::{
    DEFAULT_POSITION_D_GAIN, DEFAULT_POSITION_P_GAIN, JOINT_DEFINITIONS, JointDefinition,
};
