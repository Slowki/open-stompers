"""HighTorque motor SKUs used in the Mini Pi Plus family."""

from typing import Final, TypedDict


class HighTorqueMotor(TypedDict):
    """Joint dynamics limits of a HighTorque motor."""

    #: Peak effort in N·m.
    effort: float
    #: Maximum velocity in rad/s.
    velocity: float
    #: Joint damping in N·m·s/rad.
    damping: float


#: 5047 frame, 36:1 gearbox (legs, waist, torso).
#: https://www.hightorque.cn/wp-content/uploads/2024/06/High-Torque-Servo-Joint-MotorV2.pdf
MOTOR_5047_36: Final = HighTorqueMotor(effort=20.0, velocity=7.854, damping=0.02)

#: 4438 frame, 30:1 gearbox (arms).
#: https://hightorque.cn/en/wp-content/uploads/2023/07/User-Manual-1.pdf
MOTOR_4438_30: Final = HighTorqueMotor(effort=10.0, velocity=15.91, damping=0.02)

#: 3536 frame, 32:1 gearbox (head).
MOTOR_3536_32: Final = HighTorqueMotor(effort=3.0, velocity=30.37, damping=0.02)
