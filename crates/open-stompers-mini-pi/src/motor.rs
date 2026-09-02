//! Motor type definitions, per-motor configuration, and unit scaling helpers.
//!
//! All angular quantities use radians (position) and radians per second
//! (velocity); the on-wire representation is `int16_t` scaled by the
//! `radian-2π` convention from the reference SDK:
//!
//! - `pos_int = round(pos_rad / (2π) * 10_000)`
//! - `vel_int = round(vel_rad_per_s / (2π) * 4_000)`
//! - `tqe_int = round((torque_nm - d) / k)` with motor-specific `k`, `d`.
//! - Kp/Kd are converted through a motor-specific PID scale and then
//!   multiplied by `10 * 2π`, saturated to `±32_700`.

use core::f32::consts::TAU;

use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f32::{Angle, AngularVelocity, Torque};
use uom::si::torque::newton_meter;

/// Motor type identifiers restricted to the variants used by Mini Pi Plus.
///
/// Each variant's torque calibration `(k, d)` and PID scale match the
/// corresponding coefficients in `motor::tqe_float2int` / `motor::pid_scale`
/// in the reference SDK.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MotorType {
    /// 5047 frame, 36:1 gearbox, revised torque coefficients.
    M5047_36_2,
    /// 4438 frame, 30:1 gearbox.
    M4438_30,
    /// 3536 frame, 32:1 gearbox.
    M3536_32,
}

impl MotorType {
    /// Default position proportional gain used by the production robot model.
    #[must_use]
    pub const fn position_proportional_gain(self) -> f32 {
        match self {
            Self::M5047_36_2 => 110.0,
            Self::M4438_30 => 6.0,
            Self::M3536_32 => 6.0,
        }
    }

    /// Default velocity derivative gain used by the production robot model.
    #[must_use]
    pub const fn velocity_derivative_gain(self) -> f32 {
        match self {
            Self::M5047_36_2 => 1.1,
            Self::M4438_30 => 0.6,
            Self::M3536_32 => 0.6,
        }
    }

    /// Torque slope `k` used by the affine torque scaling.
    #[must_use]
    pub const fn torque_slope(self) -> f32 {
        match self {
            Self::M5047_36_2 => 0.008_030,
            Self::M4438_30 => 0.005_256,
            Self::M3536_32 => 0.004_581,
        }
    }

    /// Torque offset `d` used by the affine torque scaling.
    #[must_use]
    pub const fn torque_offset(self) -> f32 {
        match self {
            Self::M5047_36_2 => -0.35,
            Self::M4438_30 => -0.05,
            Self::M3536_32 => -0.105,
        }
    }

    /// Divisor applied before converting Kp/Kd to int16.
    ///
    /// Mirrors the numeric table in `motor::pid_scale` in the reference SDK.
    #[must_use]
    pub const fn pid_scale_divisor(self) -> f32 {
        match self {
            Self::M5047_36_2 => 0.803_0,
            Self::M4438_30 => 0.525_6,
            Self::M3536_32 => 0.004_581,
        }
    }
}

/// Static per-motor configuration.
#[derive(Clone, Copy, Debug)]
pub struct MotorConfig {
    /// Identifier (1..=`id_max`) that addresses this motor on the CAN bus.
    pub id: u8,
    /// Motor type used for torque and PID scaling.
    pub motor_type: MotorType,
    /// When `true`, TX position/velocity/torque are negated (and RX values
    /// negated back) so external software sees the mechanical convention.
    pub reverse: bool,
    /// Difference (radians) between the motor's encoder zero and the URDF's
    /// neutral zero for this joint: specifically, the motor-side angle the
    /// joint must be at for the URDF to consider the link at angle 0.
    ///
    /// The controller adds this value before encoding a target to the wire
    /// (so commanding zero actually moves the motor to `offset_rad`) and
    /// subtracts it again when decoding motor state responses. A joint whose
    /// mechanical zero already matches the URDF should leave this at `0.0`.
    pub offset_rad: f32,
}

impl MotorConfig {
    /// Construct a config with `reverse = false` and a zero offset.
    #[must_use]
    pub const fn new(id: u8, motor_type: MotorType) -> Self {
        Self {
            id,
            motor_type,
            reverse: false,
            offset_rad: 0.0,
        }
    }
}

/// Latest decoded motor telemetry.
///
/// Motor firmware diagnostics (mode / fault bitfields) are not cached here;
/// callers that need them should consume them directly from the RX path.
#[derive(Clone, Copy, Debug, Default)]
pub struct MotorState {
    /// Joint-side position, with `reverse` and `offset_rad` already applied.
    pub position: Angle,
    /// Joint-side velocity, with `reverse` already applied.
    pub velocity: AngularVelocity,
    /// Joint-side torque, with `reverse` already applied.
    pub torque: Torque,
}

/// Sentinel int16 used by the reference firmware to mean "leave this motor's
/// commanded position unchanged" inside a `MODE_POSITION` or
/// `MODE_POS_VEL_TQE_KP_KD` payload.
pub const POSITION_SENTINEL: i16 = -0x8000;

/// Saturation used on Kp/Kd int16 conversions to avoid overflow.
const INT16_PID_LIMIT: i32 = 32_700;

const fn saturate_i16(value: i32) -> i16 {
    #[allow(clippy::cast_possible_truncation)]
    if value >= INT16_PID_LIMIT {
        INT16_PID_LIMIT as i16
    } else if value <= -INT16_PID_LIMIT {
        -INT16_PID_LIMIT as i16
    } else {
        value as i16
    }
}

/// Convert a position to the int16 wire representation.
#[must_use]
pub fn position_to_i16(position: Angle) -> i16 {
    #[allow(clippy::cast_possible_truncation)]
    {
        (position.get::<radian>() / TAU * 10_000.0) as i16
    }
}

/// Inverse of [`position_to_i16`].
#[must_use]
pub fn position_from_i16(value: i16) -> Angle {
    Angle::new::<radian>(f32::from(value) * TAU / 10_000.0)
}

/// Convert a velocity to the int16 wire representation.
#[must_use]
pub fn velocity_to_i16(velocity: AngularVelocity) -> i16 {
    #[allow(clippy::cast_possible_truncation)]
    {
        (velocity.get::<radian_per_second>() / TAU * 4_000.0) as i16
    }
}

/// Inverse of [`velocity_to_i16`].
#[must_use]
pub fn velocity_from_i16(value: i16) -> AngularVelocity {
    AngularVelocity::new::<radian_per_second>(f32::from(value) * TAU / 4_000.0)
}

/// Convert a torque to the int16 wire representation.
#[must_use]
pub fn torque_to_i16(torque: Torque, motor_type: MotorType) -> i16 {
    #[allow(clippy::cast_possible_truncation)]
    {
        ((torque.get::<newton_meter>() - motor_type.torque_offset()) / motor_type.torque_slope())
            as i16
    }
}

/// Inverse of [`torque_to_i16`].
#[must_use]
pub fn torque_from_i16(value: i16, motor_type: MotorType) -> Torque {
    Torque::new::<newton_meter>(
        f32::from(value).mul_add(motor_type.torque_slope(), motor_type.torque_offset()),
    )
}

/// Convert a position-space gain (Kp or Kd) to the int16 wire representation.
///
/// The reference SDK uses the same formula for both — scale by the motor's
/// `pid_scale_divisor`, then multiply by `10 * 2π`, then saturate.
#[must_use]
pub const fn pid_gain_to_i16(gain: f32, motor_type: MotorType) -> i16 {
    let scaled = gain / motor_type.pid_scale_divisor();
    #[allow(clippy::cast_possible_truncation)]
    let raw = (scaled * 10.0 * TAU) as i32;
    saturate_i16(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn position_scaling_matches_reference() {
        // 2π rad → 10_000 on the wire.
        assert_eq!(position_to_i16(Angle::new::<radian>(TAU)), 10_000);
        assert_eq!(position_to_i16(Angle::new::<radian>(-TAU)), -10_000);
        assert_eq!(position_to_i16(Angle::new::<radian>(0.0)), 0);
        assert!(approx_eq(
            position_from_i16(10_000).get::<radian>(),
            TAU,
            1e-3
        ));
        assert!(approx_eq(
            position_from_i16(-5_000).get::<radian>(),
            -TAU / 2.0,
            1e-3
        ));
    }

    #[test]
    fn velocity_scaling_matches_reference() {
        assert_eq!(
            velocity_to_i16(AngularVelocity::new::<radian_per_second>(TAU)),
            4_000
        );
        assert_eq!(
            velocity_to_i16(AngularVelocity::new::<radian_per_second>(-TAU)),
            -4_000
        );
        assert!(approx_eq(
            velocity_from_i16(2_000).get::<radian_per_second>(),
            TAU / 2.0,
            1e-3
        ));
    }

    #[test]
    fn torque_roundtrip_for_each_motor_type() {
        for motor_type in [
            MotorType::M5047_36_2,
            MotorType::M4438_30,
            MotorType::M3536_32,
        ] {
            let raw = torque_to_i16(Torque::new::<newton_meter>(1.0), motor_type);
            let recovered = torque_from_i16(raw, motor_type);
            assert!(
                approx_eq(
                    recovered.get::<newton_meter>(),
                    1.0,
                    motor_type.torque_slope()
                ),
                "roundtrip failed for {motor_type:?}: got {}",
                recovered.get::<newton_meter>()
            );
        }
    }

    #[test]
    fn pid_gain_saturates_on_large_inputs() {
        assert_eq!(pid_gain_to_i16(1e6, MotorType::M5047_36_2), 32_700);
        assert_eq!(pid_gain_to_i16(-1e6, MotorType::M4438_30), -32_700);
    }

    #[test]
    fn pid_gain_zero_is_zero() {
        for motor_type in [
            MotorType::M5047_36_2,
            MotorType::M4438_30,
            MotorType::M3536_32,
        ] {
            assert_eq!(pid_gain_to_i16(0.0, motor_type), 0);
        }
    }
}
