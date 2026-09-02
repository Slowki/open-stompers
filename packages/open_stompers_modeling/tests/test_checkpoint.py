"""Shared Orbax storage, portable inference, and checkpoint validation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as checkpointing
import pytest
from etils import epath
from flax import nnx

from open_stompers_modeling.checkpoint import (
    CHECKPOINT_VERSION,
    parameter_state,
    read_checkpoint,
    read_checkpoint_metadata,
    restore_parameters,
    save_checkpoint,
)
from open_stompers_modeling.rsl.actor import MlpActor


@pytest.fixture
def actor() -> MlpActor:
    return MlpActor(3, 2, (4,), rngs=nnx.Rngs(7))


@pytest.fixture
def description() -> dict:
    return {
        "kind": MlpActor.KIND,
        "model": {"observation_size": 3, "action_size": 2, "hidden_sizes": [4]},
    }


def test_actor_output_and_host_array_round_trip(
    tmp_path: Path, actor: MlpActor, description: dict
) -> None:
    directory = tmp_path / "actor"
    parameters = parameter_state(nnx.state(actor, nnx.Param))
    save_checkpoint(directory, parameters, description)
    restored = read_checkpoint(directory, kind=MlpActor.KIND)
    assert restored["training"] is None
    assert restored["metadata"]["format"] == CHECKPOINT_VERSION
    assert "format" not in description
    assert not (directory / "training").exists()
    for expected, actual in zip(
        jax.tree.leaves(parameters), jax.tree.leaves(restored["parameters"]), strict=True
    ):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    observations = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
    snapshot = read_checkpoint(directory, kind=MlpActor.KIND)
    restored_actor = MlpActor(**snapshot["metadata"]["model"], rngs=nnx.Rngs(0))
    nnx.update(
        restored_actor,
        restore_parameters(nnx.state(restored_actor, nnx.Param), snapshot["parameters"]),
    )
    np.testing.assert_array_equal(restored_actor(observations), actor(observations))
    with pytest.raises(ValueError, match="no training state"):
        read_checkpoint(directory, include_training=True, kind=MlpActor.KIND)
    with pytest.raises(ValueError, match="dynamics_transformer"):
        read_checkpoint(directory, kind="dynamics_transformer")


def test_inference_and_metadata_do_not_restore_training(
    tmp_path: Path, actor: MlpActor, description: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "actor"
    save_checkpoint(
        directory,
        parameter_state(nnx.state(actor, nnx.Param)),
        description,
        training={"step": 4, "optimizer": np.ones(8, np.float32)},
    )
    original_restore = checkpointing.PyTreeCheckpointHandler.restore

    def restore(
        handler: checkpointing.PyTreeCheckpointHandler,
        path: epath.Path,
        *,
        args: checkpointing.args.PyTreeRestore,
    ) -> object:
        assert path.name == "parameters"
        return original_restore(handler, path, args=args)

    monkeypatch.setattr(checkpointing.PyTreeCheckpointHandler, "restore", restore)
    assert read_checkpoint_metadata(directory, kind=MlpActor.KIND)["kind"] == MlpActor.KIND
    assert read_checkpoint(directory, kind=MlpActor.KIND)["training"] is None


@pytest.mark.parametrize("damage", ["names", "shape", "dtype", "nonfinite"])
def test_parameter_validation(actor: MlpActor, damage: str) -> None:
    template = nnx.state(actor, nnx.Param)
    parameters = jax.device_get(parameter_state(template))
    match damage:
        case "names":
            parameters["unexpected"] = parameters.pop("layers")
        case "shape":
            parameters["layers"]["0"]["bias"] = np.zeros(3, np.float32)
        case "dtype":
            parameters["layers"]["0"]["bias"] = np.zeros(4, np.int32)
        case "nonfinite":
            parameters["layers"]["0"]["bias"] = np.full(4, np.nan, np.float32)
    with pytest.raises(ValueError, match="parameter"):
        restore_parameters(template, parameters)


def test_independent_orbax_writer(tmp_path: Path, actor: MlpActor, description: dict) -> None:
    directory = tmp_path / "independent"
    with checkpointing.Checkpointer(
        checkpointing.CompositeCheckpointHandler(
            parameters=checkpointing.PyTreeCheckpointHandler(),
            metadata=checkpointing.JsonCheckpointHandler(),
        )
    ) as writer:
        writer.save(
            directory,
            args=checkpointing.args.Composite(
                parameters=checkpointing.args.PyTreeSave(
                    parameter_state(nnx.state(actor, nnx.Param))
                ),
                metadata=checkpointing.args.JsonSave({**description, "format": CHECKPOINT_VERSION}),
            ),
        )
    observations = jnp.ones((2, 3))
    snapshot = read_checkpoint(directory, kind=MlpActor.KIND)
    restored_actor = MlpActor(**snapshot["metadata"]["model"], rngs=nnx.Rngs(0))
    nnx.update(
        restored_actor,
        restore_parameters(nnx.state(restored_actor, nnx.Param), snapshot["parameters"]),
    )
    np.testing.assert_array_equal(restored_actor(observations), actor(observations))


@pytest.mark.parametrize(("version", "kind"), [(5, MlpActor.KIND), (CHECKPOINT_VERSION, "unknown")])
def test_reject_unsupported_metadata(tmp_path: Path, version: int, kind: str) -> None:
    directory = tmp_path / "invalid"
    with checkpointing.Checkpointer(
        checkpointing.CompositeCheckpointHandler(
            parameters=checkpointing.PyTreeCheckpointHandler(),
            metadata=checkpointing.JsonCheckpointHandler(),
        )
    ) as writer:
        writer.save(
            directory,
            args=checkpointing.args.Composite(
                parameters=checkpointing.args.PyTreeSave({"weight": np.ones(2)}),
                metadata=checkpointing.args.JsonSave({"format": version, "kind": kind}),
            ),
        )
    with pytest.raises(ValueError, match=r"format|checkpoint"):
        read_checkpoint(directory, kind=MlpActor.KIND)


def test_reject_bare_actor_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    with checkpointing.StandardCheckpointer() as writer:
        writer.save(directory, {"layers": [{"kernel": np.ones((3, 2))}]})
        writer.wait_until_finished()
    with pytest.raises(ValueError, match="Unsupported checkpoint layout"):
        read_checkpoint(directory, kind=MlpActor.KIND)
    with pytest.raises(ValueError, match="completed Orbax"):
        read_checkpoint(tmp_path / "missing", kind=MlpActor.KIND)
