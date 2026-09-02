# Hardware

## Mini Pi Plus

Small two legged humanoid.

**Upstream Model**:

- [PiPlus_S_12L10A0G0H1W.urdf](https://github.com/HighTorque-Robotics/HT_Robot_URDF/blob/PiPlus_URDF/ht_urdf/PiPlus_S_12L10A0G0H1W/PiPlus_S_12L10A0G0H1W.urdf)
- [PiPlus_S_12L8A0G2H0W.urdf](https://github.com/HighTorque-Robotics/HT_Robot_URDF/blob/PiPlus_URDF/ht_urdf/PiPlus_S_12L8A0G2H0W/urdf/PiPlus_S_12L8A0G2H0W.urdf)
- [PiPlus_S_12L8A0G2H1W.urdf](https://github.com/HighTorque-Robotics/HT_Robot_URDF/blob/PiPlus_URDF/ht_urdf/PiPlus_S_12L8A0G2H1W/urdf/PiPlus_S_12L8A0G2H1W.urdf)

**SKUs**:

- `PiPlus_S_12L10A0G0H1W` — 23 DoF: 12 leg, 10 arm including wrists, and 1 waist; fixed head and no gripper.
- `PiPlus_S_12L8A0G2H0W` — 22 DoF: 12 leg, 8 arm with locked wrists, and 2 head; no waist or gripper.
- `PiPlus_S_12L8A0G2H1W` — 23 DoF: 12 leg, 8 arm with locked wrists, 2 head, and 1 waist; no gripper.

### Maintenance

- Use a long 2.5mm hex key for the chest bolts.
- The limb mounts require a 3mm hex key.

## Pollen Microduck

Small two legged duck-like robot.

**Upstream model (MuJoCo MJCF)**:

- [robot_walk.xml](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/robot/microduck/robot_walk.xml)
- [robot_allcollisions_rollers.xml](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/robot/microduck/robot_allcollisions_rollers.xml)

**SKUs**:

- `microduck` — 14 DoF: 10 leg and 4 neck/head DoF.
- `microduck_rollers` — 18 DoF: the 14-DoF robot plus 4 passive roller-wheel joints.

## Single model exports

Export one model to Blender, GLB, URDF, or MJCF:

```sh
uv run open_stompers_hardware_models export microduck generated/microduck.glb
uv run open_stompers_hardware_models export microduck generated/microduck.xml --format urdf
```

The format is inferred from `.blend`, `.glb`, `.urdf`, `.mjcf`, or `.mcjf`.
Destinations with another extension default to URDF. Use `--format` to override
the inferred format.
