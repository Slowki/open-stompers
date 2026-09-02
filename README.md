# Open Stompers

The goal of Open Stompers is to bring end-to-end learned autonomy to small legged robots.

## Prerequisites

- [UV](https://docs.astral.sh/uv/)
- [direnv](https://direnv.net/) (recommended)
- [FlatBuffers compiler (`flatc`)](https://flatbuffers.dev) (only required if you change any schemas)

## Set up

```sh
uv sync
uv run prek install # Install pre-commit hooks
uv run open_stompers_hardware_models fetch # Download hardware simulation assets
```

## Motion model

```mermaid
flowchart LR
    classDef node fill:#fff,stroke:#333,stroke-width:1px;
    classDef edgePaths stroke-width:2px;

    subgraph Feedback["State feedback"]
        IMU["IMU (≥ 500 Hz)"]
        MotorState["Motors State (≥ 250 Hz)"]
    end
    style Feedback fill:#fdf4ff,stroke:#e982f9,stroke-width:2px

    Locomotion["Locomotion transformer"]
    Motors["Motors"]

    Locomotion -->|"Joint Position Targets"| Motors
    linkStyle 0 stroke:#08b94e,stroke-width:2px;

    Locomotion -->|"State estimation"| Planner
    linkStyle 1 stroke:#e0ac00,stroke-width:2px;

    Planner -->|"Velocity request"| Locomotion
    linkStyle 2 stroke:#e0ac00,stroke-width:2px;
    IMU --> Locomotion
    MotorState --> Locomotion
```
