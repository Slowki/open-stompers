"""Beartype configuration used by the jaxtyping pytest hook."""

from collections.abc import Callable
from typing import Any

from beartype import BeartypeConf, beartype as _beartype


def beartype(function: Callable[..., Any]) -> Callable[..., Any]:
    """Apply beartype while accepting integers where Python permits floats."""
    return _beartype(conf=BeartypeConf(is_pep484_tower=True))(function)
