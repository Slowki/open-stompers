"""Shared Orbax checkpoint storage and portable model restoration."""

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Final, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as checkpointing
from flax import nnx, serialization

CHECKPOINT_VERSION: Final = 6


class Checkpoint(TypedDict):
    """Separated model parameters, optional training state, and metadata."""

    parameters: dict
    training: dict | None
    metadata: dict


def checkpoint_handlers() -> dict[str, checkpointing.CheckpointHandler]:
    """Create the standard Orbax item handlers."""
    return {
        "parameters": checkpointing.PyTreeCheckpointHandler(),
        "training": checkpointing.PyTreeCheckpointHandler(),
        "metadata": checkpointing.JsonCheckpointHandler(),
    }


def composite_checkpoint_handler() -> checkpointing.CompositeCheckpointHandler:
    """Create a composite handler for standalone saves and reads."""
    handlers = checkpoint_handlers()
    return checkpointing.CompositeCheckpointHandler(
        parameters=handlers["parameters"],
        training=handlers["training"],
        metadata=handlers["metadata"],
    )


def checkpoint_arguments(
    parameters: dict, metadata: dict, training: dict | None = None
) -> checkpointing.args.Composite:
    """Create isolated Orbax save arguments for a complete snapshot."""
    description = deepcopy(metadata)
    description["format"] = CHECKPOINT_VERSION
    arguments = {
        "parameters": checkpointing.args.PyTreeSave(deepcopy(parameters)),
        "metadata": checkpointing.args.JsonSave(description),
    }
    if training is not None:
        arguments["training"] = checkpointing.args.PyTreeSave(deepcopy(training))
    return checkpointing.args.Composite(**arguments)


def save_checkpoint(
    path: str | Path,
    parameters: dict,
    metadata: dict,
    training: dict | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Save a standalone checkpoint using the shared Orbax layout."""
    with checkpointing.Checkpointer(composite_checkpoint_handler()) as writer:
        writer.save(
            Path(path).resolve(),
            args=checkpoint_arguments(parameters, metadata, training),
            force=overwrite,
        )


def read_checkpoint_metadata(path: str | Path, *, kind: str | None = None) -> dict:
    """Read and validate metadata without restoring any arrays."""
    directory = Path(path).resolve()
    if not directory.is_dir() or not checkpointing.utils.is_checkpoint_finalized(directory):
        raise ValueError(f"Expected a completed Orbax checkpoint directory: {directory}")
    if not (directory / "metadata").is_dir() or not (directory / "parameters").is_dir():
        raise ValueError("Unsupported checkpoint layout; regenerate using the current format")
    with checkpointing.Checkpointer(
        checkpointing.CompositeCheckpointHandler(metadata=checkpointing.JsonCheckpointHandler())
    ) as reader:
        metadata = reader.restore(
            directory, args=checkpointing.args.Composite(metadata=checkpointing.args.JsonRestore())
        )["metadata"]
    if metadata.get("format") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Expected checkpoint format {CHECKPOINT_VERSION}; regenerate the checkpoint"
        )
    if kind is not None and metadata["kind"] != kind:
        raise ValueError(f"Expected a {kind} checkpoint")
    return metadata


def read_checkpoint(
    path: str | Path, *, include_training: bool = False, kind: str | None = None
) -> Checkpoint:
    """Restore selected items as host arrays independent of saved device sharding."""
    directory = Path(path).resolve()
    metadata = read_checkpoint_metadata(directory, kind=kind)
    names = ["parameters"]
    if include_training:
        if not (directory / "training").is_dir():
            raise ValueError("Checkpoint has no training state for resume")
        names.append("training")
    with checkpointing.Checkpointer(composite_checkpoint_handler()) as reader:
        item_metadata = reader.metadata(directory).item_metadata
        arguments = {
            name: checkpointing.args.PyTreeRestore(
                restore_args=jax.tree.map(
                    lambda leaf: (
                        checkpointing.ArrayRestoreArgs(restore_type=np.ndarray)
                        if isinstance(leaf, checkpointing.metadata.ArrayMetadata)
                        else checkpointing.RestoreArgs()
                    ),
                    item_metadata[name].tree,
                )
            )
            for name in names
        }
        restored = reader.restore(directory, args=checkpointing.args.Composite(**arguments))
    return {
        "parameters": restored["parameters"],
        "training": restored["training"] if include_training else None,
        "metadata": metadata,
    }


def parameter_state(parameters: nnx.State) -> dict:
    """Represent NNX parameters as a portable pure dictionary."""
    return serialization.to_state_dict(nnx.to_pure_dict(parameters))


def restore_parameters(template: nnx.State, parameters: dict) -> nnx.State:
    """Restore matching finite parameters while preserving NNX variable metadata."""
    expected = parameter_state(template)
    if jax.tree.structure(expected) != jax.tree.structure(parameters):
        raise ValueError("Checkpoint parameter names do not match the model")
    for original, value in zip(jax.tree.leaves(expected), jax.tree.leaves(parameters), strict=True):
        if (
            np.shape(original) != np.shape(value)
            or np.dtype(original.dtype) != np.dtype(value.dtype)
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid checkpoint parameter shape, dtype, or value")
    restored = jax.tree.map(lambda value: value, template)
    nnx.replace_by_pure_dict(restored, jax.tree.map(jnp.asarray, parameters))
    return restored


def parameter_fingerprint(parameters: dict) -> str:
    """Identify actor parameters independently of checkpoint paths."""
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(parameters)[0]:
        array = np.asarray(value)
        digest.update(repr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
