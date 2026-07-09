"""Array-backend dispatch via the Python Array API standard.

`Series.data` may be any Array-API array — numpy, torch, JAX, CuPy, Dask, …;
only these few primitives beyond plain ``__getitem__`` are needed by the
slicing algebra, and they are centralised here so every backend behaves
identically.  We lean on ``array-api-compat`` to smooth over the gaps between
each library's native namespace and the standard.

The *index* layer (``indexes.py``) is deliberately numpy-`int64` throughout —
indexes are small host-side metadata and the exact-tick arithmetic invariant
depends on it.  Only the payload array is backend-polymorphic, and it flows
through the five functions below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from array_api_compat import (
    array_namespace,
    device,
    is_array_api_obj,
    is_numpy_array,
    is_torch_array,
)


def is_tensor(x: Any) -> bool:
    """True for any Array-API array (numpy, torch, JAX, …); False for scalars.

    numpy *scalars* (``np.generic``, e.g. ``np.float64``) are treated as
    scalars, not arrays — matching the pre-existing ``isinstance(x, ndarray)``
    contract so frame entries holding numpy scalars still pass through as
    plain values rather than being wrapped as 0-d Series."""
    if isinstance(x, np.generic):
        return False
    return is_array_api_obj(x)


def _along(data: Any, axis: int, sel: Any) -> Any:
    """Apply a plain ``slice``/``int`` selector along ``axis`` — standard
    indexing, supported natively by every backend."""
    idx: list[Any] = [slice(None)] * data.ndim
    idx[axis] = sel
    return data[tuple(idx)]


def take(data: Any, axis: int, sel: Any) -> Any:
    """Select ``sel`` (slice | int | int/bool array | list) along ``axis``.

    Slices and integer scalars use standard indexing; integer/boolean arrays
    and lists go through ``xp.take`` (fancy indexing is not guaranteed by the
    standard), with the selector coerced onto the data's namespace and device.
    """
    if isinstance(sel, slice):
        return _along(data, axis, sel)
    if isinstance(sel, (int, np.integer)):
        return _along(data, axis, int(sel))
    xp = array_namespace(data)
    indices = xp.asarray(sel, device=device(data))
    if indices.dtype == xp.bool:
        indices = xp.nonzero(indices)[0]
    return xp.take(data, indices, axis=axis)


def concat(parts: list[Any], axis: int) -> Any:
    xp = array_namespace(*parts)
    return xp.concat(parts, axis=axis)


def array_equal(a: Any, b: Any) -> bool:
    """Exact equality: same backend, shape, dtype, and element-wise values.

    Arrays from different backends are never equal (matching the pre-existing
    numpy-vs-torch contract); NaNs compare unequal, as with ``np.array_equal``.
    """
    if not (is_array_api_obj(a) and is_array_api_obj(b)):
        return False
    xp_a = array_namespace(a)
    xp_b = array_namespace(b)
    if xp_a is not xp_b:
        return False
    # ``is_array_api_obj`` narrows to the array-namespace protocol, which does
    # not expose shape/dtype; the concrete arrays do.  Access them as ``Any``.
    ax: Any = a
    bx: Any = b
    if tuple(ax.shape) != tuple(bx.shape) or ax.dtype != bx.dtype:
        return False
    return bool(xp_a.all(ax == bx))


def to_numpy_f64(data: Any) -> np.ndarray:
    """Convert data to a float64 numpy array (for interpolation)."""
    if is_numpy_array(data):
        return np.asarray(data, dtype=np.float64)
    if is_torch_array(data):
        return data.detach().cpu().numpy().astype(np.float64)
    # JAX and other Array-API backends convert via the array/buffer protocol.
    return np.asarray(data, dtype=np.float64)
