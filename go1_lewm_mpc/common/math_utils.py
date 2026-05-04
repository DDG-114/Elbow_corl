"""Small numeric validation helpers used by common packet types."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def as_float_array(value: object, name: str, shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Convert a value to a float numpy array and optionally validate its shape."""
    array = np.asarray(value, dtype=np.float32)
    if shape is not None:
        validate_shape(array, shape, name)
    validate_finite(array, name)
    return array


def as_bool_array(value: object, name: str, shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Convert a value to a bool numpy array and optionally validate its shape."""
    array = np.asarray(value, dtype=bool)
    if shape is not None:
        validate_shape(array, shape, name)
    return array


def optional_float_array(
    value: object | None,
    name: str,
    shape: Optional[Sequence[int]] = None,
) -> np.ndarray | None:
    """Convert an optional value to a float array, preserving None."""
    if value is None:
        return None
    return as_float_array(value, name, shape)


def validate_shape(array: np.ndarray, expected_shape: Sequence[int | None], name: str) -> None:
    """Raise ValueError if an array shape does not match an expected shape.

    Use None in expected_shape as a wildcard for that dimension.
    """
    actual = tuple(array.shape)
    expected = tuple(expected_shape)
    if len(actual) != len(expected):
        raise ValueError(f"{name} must have shape {format_shape(expected)}, got {actual}")

    for actual_dim, expected_dim in zip(actual, expected):
        if expected_dim is not None and actual_dim != expected_dim:
            raise ValueError(f"{name} must have shape {format_shape(expected)}, got {actual}")


def validate_finite(array: np.ndarray, name: str) -> None:
    """Raise ValueError if an array contains NaN or infinite values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def validate_vector(value: object, length: int, name: str) -> np.ndarray:
    """Convert and validate a one-dimensional float vector."""
    return as_float_array(value, name, (length,))


def validate_matrix(value: object, shape: Sequence[int | None], name: str) -> np.ndarray:
    """Convert and validate a float matrix or tensor against a fixed-rank shape."""
    return as_float_array(value, name, shape)


def format_shape(shape: Iterable[int | None]) -> str:
    """Format a shape tuple for error messages."""
    return "(" + ", ".join("*" if dim is None else str(dim) for dim in shape) + ")"
