"""ONNX export helpers for modeling functions."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax2onnx


def export_onnx(
    function: Callable[..., Any],
    path: str | Path,
    inputs: Sequence[jax.ShapeDtypeStruct],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    model_name: str,
) -> Path:
    """Export a pure JAX model function to an ONNX file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jax2onnx.to_onnx(
        function,
        inputs,
        input_names=input_names,
        output_names=output_names,
        model_name=model_name,
        # We will upgrade to the newest opset after moving off JetPack 5.
        opset=21,
        return_mode="file",
        output_path=output_path,
    )
    return output_path
