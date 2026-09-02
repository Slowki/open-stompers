//! Platform-independent interfaces for robot motors, inertial sensors, and link transforms.

pub mod transforms;

pub use open_stompers_schemas::RobotPosture;

use std::future::Future;

use uom::si::f64::{Acceleration, Angle, AngularVelocity, Torque};

/// Normalized gamepad axes and the `RightThumb` mode-toggle state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GamepadCommand {
    /// Forward axis, where forward is positive and the range is -1.0 to 1.0.
    pub forward_axis: f32,
    /// Lateral axis, where left is positive and the range is -1.0 to 1.0.
    pub lateral_axis: f32,
    /// Yaw axis, where counterclockwise is positive and the range is -1.0 to 1.0.
    pub yaw_axis: f32,
    /// Whether the right thumb button is pressed.
    pub mode_toggle: bool,
}

/// Position, velocity, and optional measured torque for a motor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorState {
    /// The motor shaft angle.
    pub position: Angle,
    /// The motor shaft angular velocity.
    pub velocity: AngularVelocity,
    /// The measured output torque, when the driver can provide it.
    pub torque: Option<Torque>,
}

/// Acceleration and angular velocity measured in a sensor's local frame.
///
/// Vector elements are ordered x, y, z. Acceleration is raw accelerometer
/// output and has not had gravity removed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InertialMeasurement {
    /// Raw acceleration along the local x, y, and z axes.
    pub acceleration: [Acceleration; 3],
    /// Angular velocity along the local x, y, and z axes.
    pub angular_velocity: [AngularVelocity; 3],
}

/// IMU readings with the receive times of policy-critical data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InertialSnapshot {
    /// The latest acceleration and angular velocity values.
    pub measurement: InertialMeasurement,
    /// Scalar-first, normalized sensor orientation when available.
    pub orientation_wxyz: Option<[f64; 4]>,
    /// Receive time of the latest valid angular-velocity record.
    pub angular_velocity_received_at: Option<std::time::Instant>,
    /// Receive time of the latest valid quaternion record.
    pub orientation_received_at: Option<std::time::Instant>,
}

/// Read feedback from a motor.
pub trait Motor {
    /// The error returned by this motor driver.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Read the current motor state.
    fn read_state(&mut self) -> impl Future<Output = Result<MotorState, Self::Error>>;
}

/// Command a motor with angular position targets.
///
/// ```compile_fail
/// use open_stompers_hardware::PositionControlledMotor;
/// use uom::si::f64::Torque;
///
/// fn command<M: PositionControlledMotor>(motor: &mut M, torque: Torque) {
///     let _ = motor.set_position_target(torque, 1.0, 0.1);
/// }
/// ```
pub trait PositionControlledMotor: Motor {
    /// Set the desired angular position.
    fn set_position_target(
        &mut self,
        target: Angle,
        kp: f64,
        kd: f64,
    ) -> impl Future<Output = Result<(), Self::Error>>;
}

/// Command a motor with angular velocity targets.
pub trait VelocityControlledMotor: Motor {
    /// Set the desired angular velocity.
    fn set_velocity_target(
        &mut self,
        target: AngularVelocity,
    ) -> impl Future<Output = Result<(), Self::Error>>;
}

/// Command a motor with torque targets.
pub trait TorqueControlledMotor: Motor {
    /// Set the desired torque.
    fn set_torque_target(
        &mut self,
        target: Torque,
    ) -> impl Future<Output = Result<(), Self::Error>>;
}

/// Read acceleration and angular velocity from an inertial sensor.
pub trait InertialMeasurementSensor {
    /// The error returned by this sensor driver.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Read the current inertial measurement.
    fn read_measurement(
        &mut self,
    ) -> impl Future<Output = Result<InertialMeasurement, Self::Error>>;
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    use uom::si::acceleration::meter_per_second_squared;
    use uom::si::angle::degree;
    use uom::si::angular_velocity::radian_per_second;
    use uom::si::f64::{Acceleration, Angle, AngularVelocity, Torque};
    use uom::si::torque::newton_meter;

    use super::{
        InertialMeasurement, InertialMeasurementSensor, Motor, MotorState, PositionControlledMotor,
        TorqueControlledMotor, VelocityControlledMotor,
    };

    #[derive(Debug, PartialEq)]
    struct DriverError;

    impl std::fmt::Display for DriverError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "driver failure")
        }
    }

    impl std::error::Error for DriverError {}

    #[derive(Default)]
    struct FakeMotor {
        state: Option<MotorState>,
        target: Option<Target>,
        fail: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Target {
        Position { target: Angle, kp: f64, kd: f64 },
        Velocity(AngularVelocity),
        Torque(Torque),
    }

    impl Motor for FakeMotor {
        type Error = DriverError;

        fn read_state(&mut self) -> impl Future<Output = Result<MotorState, Self::Error>> {
            std::future::ready(if self.fail {
                Err(DriverError)
            } else {
                Ok(self.state.expect("test motor state is set"))
            })
        }
    }

    impl PositionControlledMotor for FakeMotor {
        fn set_position_target(
            &mut self,
            target: Angle,
            kp: f64,
            kd: f64,
        ) -> impl Future<Output = Result<(), Self::Error>> {
            self.target = Some(Target::Position { target, kp, kd });
            std::future::ready(if self.fail { Err(DriverError) } else { Ok(()) })
        }
    }

    impl VelocityControlledMotor for FakeMotor {
        fn set_velocity_target(
            &mut self,
            target: AngularVelocity,
        ) -> impl Future<Output = Result<(), Self::Error>> {
            self.target = Some(Target::Velocity(target));
            std::future::ready(if self.fail { Err(DriverError) } else { Ok(()) })
        }
    }

    impl TorqueControlledMotor for FakeMotor {
        fn set_torque_target(
            &mut self,
            target: Torque,
        ) -> impl Future<Output = Result<(), Self::Error>> {
            self.target = Some(Target::Torque(target));
            std::future::ready(if self.fail { Err(DriverError) } else { Ok(()) })
        }
    }

    #[derive(Default)]
    struct FakeSensor {
        measurement: Option<InertialMeasurement>,
        fail: bool,
    }

    impl InertialMeasurementSensor for FakeSensor {
        type Error = DriverError;

        fn read_measurement(
            &mut self,
        ) -> impl Future<Output = Result<InertialMeasurement, Self::Error>> {
            std::future::ready(if self.fail {
                Err(DriverError)
            } else {
                Ok(self.measurement.expect("test sensor measurement is set"))
            })
        }
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut future = pin!(future);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("test futures complete on their first poll"),
        }
    }

    #[test]
    fn motor_reads_state_with_unit_conversions() {
        let expected = MotorState {
            position: Angle::new::<degree>(90.0),
            velocity: AngularVelocity::new::<radian_per_second>(2.0),
            torque: Some(Torque::new::<newton_meter>(1.5)),
        };
        let mut motor = FakeMotor {
            state: Some(expected),
            ..FakeMotor::default()
        };

        let actual = block_on(motor.read_state());

        assert_eq!(actual, Ok(expected));
        assert!(
            (expected.position.get::<uom::si::angle::radian>() - std::f64::consts::FRAC_PI_2).abs()
                < 1e-12
        );
    }

    #[test]
    fn motor_sets_position_velocity_and_torque_targets() {
        let mut motor = FakeMotor::default();
        let position = Angle::new::<degree>(30.0);
        let kp = 10.5;
        let kd = 0.25;
        assert_eq!(
            block_on(motor.set_position_target(position, kp, kd)),
            Ok(())
        );
        assert_eq!(
            motor.target,
            Some(Target::Position {
                target: position,
                kp,
                kd
            })
        );

        let velocity = AngularVelocity::new::<radian_per_second>(3.0);
        assert_eq!(block_on(motor.set_velocity_target(velocity)), Ok(()));
        assert_eq!(motor.target, Some(Target::Velocity(velocity)));

        let torque = Torque::new::<newton_meter>(0.5);
        assert_eq!(block_on(motor.set_torque_target(torque)), Ok(()));
        assert_eq!(motor.target, Some(Target::Torque(torque)));
    }

    #[test]
    fn motor_propagates_driver_errors() {
        let mut motor = FakeMotor {
            fail: true,
            ..FakeMotor::default()
        };
        assert_eq!(
            block_on(motor.set_position_target(Angle::new::<degree>(1.0), 1.0, 0.1)),
            Err(DriverError)
        );
    }

    #[test]
    fn sensor_reads_local_axes_and_propagates_driver_errors() {
        let measurement = InertialMeasurement {
            acceleration: [
                Acceleration::new::<meter_per_second_squared>(1.0),
                Acceleration::new::<meter_per_second_squared>(2.0),
                Acceleration::new::<meter_per_second_squared>(3.0),
            ],
            angular_velocity: [
                AngularVelocity::new::<radian_per_second>(4.0),
                AngularVelocity::new::<radian_per_second>(5.0),
                AngularVelocity::new::<radian_per_second>(6.0),
            ],
        };
        let mut sensor = FakeSensor {
            measurement: Some(measurement),
            ..FakeSensor::default()
        };
        assert_eq!(block_on(sensor.read_measurement()), Ok(measurement));

        sensor.fail = true;
        assert_eq!(block_on(sensor.read_measurement()), Err(DriverError));
    }
}
