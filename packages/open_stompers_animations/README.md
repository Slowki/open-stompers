# Open Stompers Animations

Procedural loops and actions for the robots in `open_stompers_hardware_models`.

## Usage

```bash
uv run animations list
uv run animations generate walk --robot PiPlus_S_12L8A0G2H1W --velocity 0.25 --cadence 100 --output generated/animations/ --analyze
uv run animations library --robot microduck --output library/microduck
uv run animations view generated/animations/*.parquet
uv run animations export-blender generated/animations/walk.parquet walk.blend
```

**Loops** play one seamless gait cycle:

- `walk`, `jog`, `run`
- `turn_in_place`, `turn_while_walking`
- `side_step`

**Actions** play once:

- `start_walking`, `stop_walking`
- `walk_to_run`, `run_to_walk`, `run_to_stop`
- `sit_down`, `stand_up`
- `jump_vertical`

## Cadence

Unless a gait in `robots.py` sets one, cadence is the average human cadence at the
dynamically similar speed, scaled by the robot's standing height (data and sources in
`planning/cadence.py`):

- Walking uses the human walk ratio, step length / cadence ≈ 0.0064 m per step/min, so
  `cadence = sqrt(60 v / 0.0064)`: about 110 steps/min at 1.3 m/s.
- Running uses about 160 steps/min at 2.7 m/s, plus 9 steps/min per m/s, for runners
  1.75 m tall, and never below 150 steps/min.
- A robot of height `h` at speed `v` matches a human at `v / sqrt(h / 1.75)` (equal Froude
  number), and its cadence is the human cadence times `sqrt(1.75 / h)`.

## Format

Each animation is one Parquet file with one row per frame:

| Column              | Shape | Meaning                                                 |
| ------------------- | ----- | ------------------------------------------------------- |
| `time`              | ()    | Seconds.                                                |
| `root_position`     | (3)   | Meters.                                                 |
| `root_quaternion`   | (4)   | WXYZ.                                                   |
| `joint_position`    | (J)   | Radians.                                                |
| `command`           | (3)   | Policy label `[vx, vy, yaw_rate]` in the heading frame. |
| `behavior`          | ()    | Discrete label: `walk`, `run`, `sit`, …                 |
| `phase`             | ()    | Gait phase in [0, 1).                                   |
| `contact`           | (3)   | Left foot, right foot, support pad.                     |
| `center_of_mass`    | (3)   | Planned whole-body center of mass.                      |
| `zero_moment_point` | (2)   | Planned zero moment point.                              |

The schema metadata key `open_stompers_animation` holds the header:

- joint and contact names
- sample interval and kind
- the loop offset, a planar transform from one cycle to the next
- the generator parameters, including the cadence actually used

`sample_animation` evaluates any time, tiling loops; `resample` changes the rate; and
`conversion.to_motion_clip` produces a datasets `MotionClip`.

## Method

1. **Footsteps.** The command schedule is integrated into a heading path. Feet land beside
   the path at mid-stance, so a loop's average velocity equals its command exactly. Each
   gait sets its own step width; sideways steps widen it only to keep the feet apart.
2. **Height.** The solver first finds, frame by frame, the tallest pose whose planted knees
   bend by the gait's `stance_knee_flexion` within every joint limit. A least-squares planner
   smooths those heights, enforces free fall during flight phases, and lifts swing feet with
   the body.
3. **Horizontal center of mass.** A batch zero moment point preview controller uses the
   variable-height pendulum balance `(g + z̈)(c − p) = z c̈`. It treats walking, running,
   and gait transitions alike, with periodic boundaries for loops.
4. **Inverse kinematics.** Whole-body damped least squares in MuJoCo tracks the soles, the
   center of mass, and the base orientation. Feet roll onto their toes before liftoff.
5. **Posture and arms.** Jogging and running lean the trunk forward, and transitions add a
   lean of `atan(a / g)` into their acceleration. Arms swing opposite the legs with a gain
   fitted per gait to cancel the legs' vertical angular momentum, within human-like swing
   limits; jogging and running bend the elbows and carry the arms forward.

## Analysis

`analyze` writes `<name>.analysis.json` and `<name>.analysis.parquet`. Per frame, it computes:

- `mj_inverse` generalized forces;
- contact forces in linearized friction cones, solved with non-negative least squares;
- joint torques, center of pressure, support margin, and foot slip;
- self-collision depth and the realized center of mass.

The summary reports:

- peak and RMS torque per joint against the effort limits, plus speed against the speed limits
- mechanical power and cost of transport
- required friction
- center of mass height, sway, and deviation from the plan
- mean and least knee bend of loaded legs
- the root force the contacts cannot supply

That last value is the dynamic infeasibility. It is split into steady frames and frames
within one sample of a contact change, because impacts are not resolvable at 50 Hz.

## Tuning sway, ankle load, and leg extension

These pull against each other, and the per-robot settings in `robots.py` balance them:

- Straighter legs raise the center of mass, and a taller inverted pendulum sways more.
- Narrower steps (`step_width`) reduce sway, but Mini Pi Plus hips adduct only 0.17 rad.
- Moving the zero moment point toward the body (`support_inset`) reduces sway at the cost of
  ankle roll torque.
- Faster cadence reduces sway the most.

## Known limits

- Mini Pi Plus running needs above the full 20 N·m knee rating.
- The Microduck support pad cannot reach the floor with flat feet, so its `sit_down` is a
  deep squat with no pad contact.
