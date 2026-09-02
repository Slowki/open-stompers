//! Platform-specific Mini Pi Plus motor and IMU drivers.

mod background;

pub mod config;
pub mod controller;
pub mod crc;
pub mod error;
pub mod mini_pi_plus;
pub mod motor;
pub mod protocol;
pub mod rsl_mlp_amp_inference;
pub mod transforms;
pub mod velocity_policy;
pub mod yesense_driver;
pub mod yesense_protocol;

pub use background::ReceivedMotorState;
pub use config::{
    DEFAULT_IMU_SERIAL_PATH, DEFAULT_MOTOR_BAUDRATE, DEFAULT_MOTOR_SERIAL_PATHS, MiniPiPlusConfig,
    SERIAL_PORT_COUNT,
};
pub use controller::{FunVersion, MotorController};
pub use error::{Error, Result};
pub use mini_pi_plus::{
    HEAD_POSITION_DERIVATIVE_GAIN, HEAD_POSITION_PROPORTIONAL_GAIN, JOINT_DEFINITIONS,
    JointDefinition, MiniPiPlus, MiniPiPlusMotor, WAIST_POSITION_DERIVATIVE_GAIN,
    WAIST_POSITION_PROPORTIONAL_GAIN,
};
pub use motor::MotorState;
pub use motor::{MotorConfig, MotorType};
pub use yesense_driver::{DEFAULT_YESENSE_BAUDRATE, YesenseDriver};
