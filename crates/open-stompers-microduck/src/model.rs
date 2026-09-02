//! Microduck joint topology and XL330 unit conversions.
//!
//! Joint names follow the URDF. Dynamixel IDs and the position conversion match the
//! alpha robot bus: left leg, neck and head, mouth, then right leg.

use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{Angle, AngularVelocity, Torque};
use uom::si::torque::newton_meter;

/// Number of Dynamixel servos on the Microduck bus, including the mouth.
pub const JOINT_COUNT: usize = 15;

/// Dynamixel ID of the `imu_to_dxl` board. It shares the servo bus.
pub const IMU_ID: u8 = 200;

/// XL330 position P gain written by the robot's control loop.
pub const DEFAULT_POSITION_P_GAIN: f32 = 200.0;

/// XL330 position D gain written by the robot's control loop.
pub const DEFAULT_POSITION_D_GAIN: f32 = 0.0;

/// XL330-M288 torque constant at 5 V, from the stall rating `0.354 N·m/A`.
///
/// Present current is one milliamp per count. This converts that current into
/// shaft torque for [`open_stompers_hardware::MotorState`].
pub const XL330_M288_NEWTON_METERS_PER_AMPERE: f64 = 0.354;

/// `0.229 rev/min` per present-velocity count, in radians per second.
const RADIANS_PER_SECOND_PER_COUNT: f64 = 0.229 * (2.0 * std::f64::consts::PI / 60.0);

/// Static Microduck joint-to-servo wiring description.
#[derive(Clone, Copy, Debug)]
pub struct JointDefinition {
    /// URDF joint name.
    pub name: &'static str,
    /// Dynamixel ID.
    pub id: u8,
    /// Position P gain used by the robot control loop, as an XL330 register value.
    pub proportional_gain: f32,
    /// Position D gain used by the robot control loop, as an XL330 register value.
    pub derivative_gain: f32,
}

const fn joint(name: &'static str, id: u8) -> JointDefinition {
    JointDefinition {
        name,
        id,
        proportional_gain: DEFAULT_POSITION_P_GAIN,
        derivative_gain: DEFAULT_POSITION_D_GAIN,
    }
}

/// Servo topology for the alpha Microduck.
///
/// The mouth is a real servo. The kinematic model used for visualization has no
/// independent mouth joint, so that name is absent from the generated transforms.
pub const JOINT_DEFINITIONS: [JointDefinition; JOINT_COUNT] = [
    joint("left_hip_yaw", 20),
    joint("left_hip_roll", 21),
    joint("left_hip_pitch", 22),
    joint("left_knee", 23),
    joint("left_ankle", 24),
    joint("neck_pitch", 30),
    joint("head_pitch", 31),
    joint("head_yaw", 32),
    joint("head_roll", 33),
    joint("mouth", 34),
    joint("right_hip_yaw", 10),
    joint("right_hip_roll", 11),
    joint("right_hip_pitch", 12),
    joint("right_knee", 13),
    joint("right_ankle", 14),
];

/// Convert an XL330 present-position count to a shaft angle.
///
/// `2048` is zero. The servo applies `homing_offset` before reporting this count.
#[must_use]
pub fn position_from_raw(raw: i32) -> Angle {
    Angle::new::<radian>(
        (2.0 * std::f64::consts::PI * f64::from(raw) / 4096.0) - std::f64::consts::PI,
    )
}

/// Convert a shaft angle to an XL330 goal-position count.
#[must_use]
pub fn position_to_raw(position: Angle) -> i32 {
    let radians = position.get::<radian>();
    #[allow(clippy::cast_possible_truncation)]
    {
        (4096.0 * (std::f64::consts::PI + radians) / (2.0 * std::f64::consts::PI)) as i32
    }
}

/// Convert an XL330 present-velocity count to a shaft rate.
#[must_use]
pub fn velocity_from_raw(raw: i32) -> AngularVelocity {
    AngularVelocity::new::<radian_per_second>(f64::from(raw) * RADIANS_PER_SECOND_PER_COUNT)
}

/// Convert an XL330 present-current count to shaft torque.
#[must_use]
pub fn torque_from_raw(raw: i16) -> Torque {
    Torque::new::<newton_meter>(f64::from(raw) * XL330_M288_NEWTON_METERS_PER_AMPERE / 1_000.0)
}

/// IDs read on every telemetry transaction, IMU board first.
#[must_use]
pub fn telemetry_ids() -> Vec<u8> {
    let mut ids = Vec::with_capacity(JOINT_COUNT + 1);
    ids.push(IMU_ID);
    ids.extend(JOINT_DEFINITIONS.iter().map(|joint| joint.id));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_ids_and_names_are_unique_and_leave_the_imu_free() {
        let mut names = std::collections::HashSet::new();
        let mut ids = std::collections::HashSet::new();
        for joint in JOINT_DEFINITIONS {
            assert!(names.insert(joint.name));
            assert!(ids.insert(joint.id));
            assert_ne!(joint.id, IMU_ID);
            assert_ne!(joint.id, 1, "id 1 is the factory id of a replacement servo");
        }
        assert_eq!(JOINT_DEFINITIONS.len(), JOINT_COUNT);
    }

    #[test]
    fn position_conversion_round_trips_center_and_endpoints() {
        for raw in [0, 1024, 2048, 3072, 4095] {
            let angle = position_from_raw(raw);
            assert_eq!(position_to_raw(angle), raw);
        }
        assert!(position_from_raw(2048).get::<radian>().abs() < 1e-12);
    }

    #[test]
    fn velocity_scale_matches_the_datasheet_figure() {
        let one_count = velocity_from_raw(1).get::<radian_per_second>();
        assert!((one_count * 60.0 / (2.0 * std::f64::consts::PI) - 0.229).abs() < 1e-12);
    }
}
