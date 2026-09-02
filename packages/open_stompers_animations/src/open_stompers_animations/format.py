"""Parquet storage of animations with one row per frame."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from .types import SCHEMA_VERSION, Animation, AnimationKind, Behavior, LoopOffset

#: Parquet schema metadata key holding the animation header.
METADATA_KEY: Final = b"open_stompers_animation"

_VECTOR_COLUMNS: Final = (
    ("root_position", "root_positions"),
    ("root_quaternion", "root_quaternions"),
    ("joint_position", "joint_positions"),
    ("command", "commands"),
    ("center_of_mass", "center_of_mass"),
    ("zero_moment_point", "zero_moment_point"),
)


def _fixed_size_column(values: NDArray) -> pa.FixedSizeListArray:
    """Convert a (frames, width) array into a fixed-size list column."""
    flat = pa.array(np.ascontiguousarray(values).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _column_values(table: pa.Table, name: str, dtype: type) -> NDArray:
    """Read a fixed-size list column back into a (frames, width) array."""
    column = table.column(name).combine_chunks()
    width = column.type.list_size
    return column.flatten().to_numpy(zero_copy_only=False).astype(dtype).reshape(-1, width)


def animation_table(animation: Animation) -> pa.Table:
    """Build the Parquet table of an animation, including its schema metadata.

    Args:
        animation: Animation to encode.

    Returns:
        One row per frame with the header stored in the schema metadata.
    """
    columns: dict[str, pa.Array] = {
        "time": pa.array(animation.times.astype(np.float32)),
        "behavior": pa.array(
            [str(behavior) for behavior in animation.behaviors]
        ).dictionary_encode(),
        "phase": pa.array(animation.phases),
        "contact": _fixed_size_column(animation.contacts),
    }
    for column, attribute in _VECTOR_COLUMNS:
        columns[column] = _fixed_size_column(getattr(animation, attribute))
    header = {
        "schema_version": SCHEMA_VERSION,
        "name": animation.name,
        "variant": animation.variant,
        "kind": str(animation.kind),
        "sample_interval": animation.sample_interval,
        "joint_names": list(animation.joint_names),
        "contact_names": list(animation.contact_names),
        "loop_offset": asdict(animation.loop_offset),
        "parameters": animation.parameters,
        "metadata": animation.metadata,
    }
    table = pa.table(columns)
    return table.replace_schema_metadata({METADATA_KEY: json.dumps(header).encode()})


def write_animation(animation: Animation, path: str | Path) -> Path:
    """Write an animation to a Parquet file.

    Args:
        animation: Animation to store.
        path: Destination file.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    pq.write_table(animation_table(animation), temporary, compression="zstd")
    temporary.replace(destination)
    return destination


def read_animation(path: str | Path) -> Animation:
    """Read an animation written by `write_animation`.

    Args:
        path: Parquet file to read.

    Returns:
        The decoded animation.

    Raises:
        ValueError: If the file is not a supported animation.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if METADATA_KEY not in metadata:
        raise ValueError(f"{path} is not an Open Stompers animation")
    header = json.loads(metadata[METADATA_KEY])
    if header["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported animation schema version {header['schema_version']}")
    vectors = {
        attribute: _column_values(table, column, np.float32)
        for column, attribute in _VECTOR_COLUMNS
    }
    return Animation(
        name=header["name"],
        variant=header["variant"],
        kind=AnimationKind(header["kind"]),
        sample_interval=float(header["sample_interval"]),
        joint_names=tuple(header["joint_names"]),
        behaviors=tuple(
            Behavior(value) for value in table.column("behavior").cast(pa.string()).to_pylist()
        ),
        phases=table.column("phase").to_numpy().astype(np.float32),
        contact_names=tuple(header["contact_names"]),
        contacts=_column_values(table, "contact", bool),
        loop_offset=LoopOffset(**header["loop_offset"]),
        parameters=header["parameters"],
        metadata=header["metadata"],
        **vectors,
    )


__all__ = ["METADATA_KEY", "animation_table", "read_animation", "write_animation"]
