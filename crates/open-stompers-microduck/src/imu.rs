//! `imu_to_dxl` v2 block decoder.
//!
//! The board is read in the same 12-byte transaction as the servos. Those bytes are
//! gyro counts and an SFLP game-rotation quaternion, not a raw accelerometer sample.
//! Acceleration reported through [`open_stompers_hardware::InertialMeasurement`] is
//! therefore the specific force of the fused orientation, with gravity left in.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use open_stompers_hardware::InertialMeasurement;
use serde::Deserialize;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{Acceleration, AngularVelocity};

use crate::protocol::PRESENT_BLOCK_LEN;

/// Bytes consumed from the IMU board on each telemetry read.
pub const IMU_BLOCK_LEN: usize = PRESENT_BLOCK_LEN as usize;

/// Standard gravity, used to turn the fused gravity direction into specific force.
const STANDARD_GRAVITY: f64 = 9.80665;

/// ±500 dps at 17.5 mdps per count, in radians per second.
const GYRO_RADIANS_PER_COUNT: f64 = 0.0175 * std::f64::consts::PI / 180.0;

/// How the IMU board is mounted relative to the trunk.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ImuMount {
    /// `zero3` mount: trunk axes are `[+sensor_z, +sensor_y, −sensor_x]`.
    #[default]
    Zero3,
    /// `beta` mount: the board axes already match the trunk.
    Beta,
}

impl ImuMount {
    /// Sensor-to-trunk rotation, scalar-first `[w, x, y, z]`.
    #[must_use]
    pub const fn quaternion_wxyz(self) -> [f64; 4] {
        match self {
            Self::Zero3 => [
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ],
            Self::Beta => [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn rotation(self) -> UnitQuaternion<f64> {
        let [w, x, y, z] = self.quaternion_wxyz();
        UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z))
    }
}

/// Stateful decoder for one IMU board.
#[derive(Clone, Debug)]
pub struct ImuDecoder {
    mount: UnitQuaternion<f64>,
    /// Last quaternion the board produced, trunk to world, scalar-first.
    orientation_wxyz: Option<[f64; 4]>,
}

impl ImuDecoder {
    /// Decode subsequent blocks with `mount`.
    #[must_use]
    pub fn new(mount: ImuMount) -> Self {
        Self {
            mount: mount.rotation(),
            orientation_wxyz: None,
        }
    }

    /// Decode one 12-byte block.
    ///
    /// An all-zero quaternion means the chip has not written its SFLP table yet.
    /// Gyro is still reported, and the previous orientation is held.
    #[must_use]
    pub fn decode(&mut self, raw: &[u8; IMU_BLOCK_LEN]) -> DecodedImu {
        let sensor_gyro = Vector3::new(
            f64::from(i16::from_le_bytes([raw[0], raw[1]])) * GYRO_RADIANS_PER_COUNT,
            f64::from(i16::from_le_bytes([raw[2], raw[3]])) * GYRO_RADIANS_PER_COUNT,
            f64::from(i16::from_le_bytes([raw[4], raw[5]])) * GYRO_RADIANS_PER_COUNT,
        );
        let gyro = self.mount.transform_vector(&sensor_gyro);

        let packed = [
            u16::from_le_bytes([raw[6], raw[7]]),
            u16::from_le_bytes([raw[8], raw[9]]),
            u16::from_le_bytes([raw[10], raw[11]]),
        ];
        if packed != [0, 0, 0] {
            let (x, y, z) = (half(packed[0]), half(packed[1]), half(packed[2]));
            let norm_squared = x * x + y * y + z * z;
            if x.is_finite() && y.is_finite() && z.is_finite() && norm_squared <= 1.02 {
                let w = (1.0 - norm_squared).max(0.0).sqrt();
                let sensor = UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z));
                let trunk = sensor * self.mount.inverse();
                let coordinates = trunk.coords;
                self.orientation_wxyz =
                    Some([coordinates.w, coordinates.x, coordinates.y, coordinates.z]);
            }
        }

        let acceleration = self.orientation_wxyz.map_or([0.0; 3], |orientation| {
            let orientation = UnitQuaternion::new_normalize(Quaternion::new(
                orientation[0],
                orientation[1],
                orientation[2],
                orientation[3],
            ));
            let gravity = orientation
                .inverse()
                .transform_vector(&Vector3::new(0.0, 0.0, -1.0));
            let magnitude = gravity.norm();
            if magnitude > 0.1 {
                let specific_force = -gravity / magnitude * STANDARD_GRAVITY;
                [specific_force.x, specific_force.y, specific_force.z]
            } else {
                [0.0; 3]
            }
        });

        DecodedImu {
            measurement: InertialMeasurement {
                acceleration: acceleration.map(Acceleration::new::<meter_per_second_squared>),
                angular_velocity: [gyro.x, gyro.y, gyro.z]
                    .map(AngularVelocity::new::<radian_per_second>),
            },
            orientation_wxyz: self.orientation_wxyz,
        }
    }
}

/// One decoded IMU sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodedImu {
    /// Gyro in the trunk frame, and specific force estimated from orientation.
    pub measurement: InertialMeasurement,
    /// Trunk-to-world quaternion, scalar-first, once the chip has produced one.
    pub orientation_wxyz: Option<[f64; 4]>,
}

fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let fraction = f64::from(bits & 0x3FF);
    match exponent {
        0 => sign * fraction * 2.0_f64.powi(-24),
        0x1F if fraction == 0.0 => sign * f64::INFINITY,
        0x1F => f64::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2.0_f64.powi(exponent - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero3_mount_matches_the_board_axis_map() {
        let mount = ImuMount::Zero3.rotation();
        let x = mount.transform_vector(&Vector3::x());
        let y = mount.transform_vector(&Vector3::y());
        let z = mount.transform_vector(&Vector3::z());
        assert!(x.metric_distance(&Vector3::new(0.0, 0.0, -1.0)) < 1e-12);
        assert!(y.metric_distance(&Vector3::y()) < 1e-12);
        assert!(z.metric_distance(&Vector3::x()) < 1e-12);
    }

    #[test]
    fn gyro_counts_are_signed_and_zero_quaternion_holds_orientation() {
        let mut decoder = ImuDecoder::new(ImuMount::Beta);
        let mut block = [0_u8; IMU_BLOCK_LEN];
        block[..2].copy_from_slice(&(-1_i16).to_le_bytes());
        let decoded = decoder.decode(&block);
        let expected = -GYRO_RADIANS_PER_COUNT;
        assert!(
            (decoded.measurement.angular_velocity[0].get::<radian_per_second>() - expected).abs()
                < 1e-12
        );
        assert!(decoded.orientation_wxyz.is_none());
        assert_eq!(
            decoded.measurement.acceleration[2].get::<meter_per_second_squared>(),
            0.0
        );

        // Half-precision 0.0 is all zeros, so use a small x component (0x2E66 ≈ 0.1).
        block[6..8].copy_from_slice(&0x2E66_u16.to_le_bytes());
        let decoded = decoder.decode(&block);
        let orientation = decoded.orientation_wxyz.expect("quaternion");
        assert!(orientation[0] > 0.9);
        assert!(
            (decoded.measurement.acceleration[2].get::<meter_per_second_squared>()
                - STANDARD_GRAVITY)
                .abs()
                < 0.2
        );

        block[6..8].copy_from_slice(&0_u16.to_le_bytes());
        let held = decoder.decode(&block);
        assert_eq!(held.orientation_wxyz, Some(orientation));
    }

    #[test]
    fn half_precision_decodes_common_values() {
        assert_eq!(half(0x3C00), 1.0);
        assert_eq!(half(0xC000), -2.0);
        assert_eq!(half(0), 0.0);
    }
}
