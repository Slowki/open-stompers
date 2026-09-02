# Open Stompers modeling

Neural network modeling for Open Stompers.

## Velocity actor

`VelocityActor` accepts physical sensor/action history, a valid-token mask, and desired base-frame `[forward_velocity, lateral_velocity, yaw_rate]`. It returns Gaussian residual-action means and deviations.
Select a compact transformer or a short-history perceptron through `ActorSettings`.

Each token contains gyro, projected gravity, neutral-relative joint positions, velocity-limit-scaled joint velocities, and the previous sampled action. Its width is `6 + 3 * joints`. `ObservationContract` records joint ordering, physical scaling, motor limits, gains, and timing.
Invalid history padding cannot affect actor output.
