"""Minimal array-backend dispatch.

`Series.data` may be a numpy array or a torch tensor.  The slicing algebra
only needs four primitives beyond plain ``__getitem__``; they are centralised
here so both backends behave identically.  torch is an optional dependency —
it is imported lazily and only when a torch tensor is actually encountered.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch():
    import torch

    return torch


def is_tensor(x: Any) -> bool:
    """True for numpy arrays and torch tensors."""
    if isinstance(x, np.ndarray):
        return True
    mod = type(x).__module__
    return mod is not None and mod.split(".")[0] == "torch"


def take(data: Any, axis: int, sel: Any) -> Any:
    """Select ``sel`` (slice | int | int/bool array | list) along ``axis``."""
    if isinstance(sel, np.ndarray) and not isinstance(data, np.ndarray):
        sel = _torch().as_tensor(sel)
    idx: list[Any] = [slice(None)] * data.ndim
    idx[axis] = sel
    return data[tuple(idx)]


def concat(parts: list[Any], axis: int) -> Any:
    if isinstance(parts[0], np.ndarray):
        return np.concatenate(parts, axis=axis)
    return _torch().cat(parts, dim=axis)


def array_equal(a: Any, b: Any) -> bool:
    a_np = isinstance(a, np.ndarray)
    b_np = isinstance(b, np.ndarray)
    if a_np != b_np:
        return False
    if a_np:
        return bool(np.array_equal(a, b))
    torch = _torch()
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def to_numpy_f64(data: Any) -> np.ndarray:
    """Convert data to a float64 numpy array (for interpolation)."""
    if isinstance(data, np.ndarray):
        return np.asarray(data, dtype=np.float64)
    return data.detach().cpu().numpy().astype(np.float64)
