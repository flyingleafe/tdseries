"""Backend-parametrized invariants for the payload array.

``Series.data`` is dispatched through the Array-API standard (``_array.py``),
so the slice/concat identity and structural equality must hold identically
whether the payload is numpy, torch, or JAX.  The *index* layer stays numpy-
`int64` regardless of payload backend (DESIGN invariant 2), so these tests
convert only ``data`` and lean on the numpy strategies for everything else.

JAX and torch are optional; a backend that is not importable is simply dropped
from the parameter list (numpy is always present).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from array_api_compat import is_jax_array, is_numpy_array, is_torch_array
from hypothesis import given, settings
from hypothesis import strategies as st

from tdseries.series import Series

from .strategies import cut_points_ticks, multi_mic_uniform_series, uniform_series

# This module proves the payload dispatch layer (_array.py) behaves identically
# across backends; the exhaustive tick/shape fuzzing already lives in the
# numpy-only suites at 100-200 examples.  A small budget on small arrays is
# enough here and keeps the JAX/XLA compilation cache (which grows per distinct
# shape) within the memory of a modest CI runner.
_EX = 20
_MAX_N = 16


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


BACKENDS = ["numpy"]
if _available("torch"):
    BACKENDS.append("torch")
if _available("jax"):
    import jax

    # Keep float64 payloads exact (JAX silently downcasts to float32 otherwise);
    # the identity tests do no arithmetic on data, but exactness keeps dtypes
    # aligned with the numpy strategies.
    jax.config.update("jax_enable_x64", True)
    BACKENDS.append("jax")


def _to_backend(s: Series, backend: str) -> Series:
    """Re-wrap ``s`` with its payload moved onto ``backend`` (indexes untouched)."""
    if s.data is None or backend == "numpy":
        return s
    base = np.asarray(s.data)
    if backend == "torch":
        import torch

        return s.with_data(torch.as_tensor(base))
    if backend == "jax":
        import jax.numpy as jnp

        return s.with_data(jnp.asarray(base))
    raise AssertionError(backend)


def _is_native(data: object, backend: str) -> bool:
    checker = {"numpy": is_numpy_array, "torch": is_torch_array, "jax": is_jax_array}[backend]
    return checker(data)


@st.composite
def _cuts(draw, s: Series, k: int) -> list[int]:
    return draw(cut_points_ticks(s.t_start_ticks, s.t_end_ticks, k))


# ---------------------------------------------------------------------------
# The load-bearing invariant, per backend:  x[a:b] ++ x[b:c] == x[a:c]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(uniform_series(max_n=_MAX_N), st.data())
def test_slice_concat_identity(backend: str, us: Series, data):
    us = _to_backend(us, backend)
    a, b, c = data.draw(_cuts(us, 3))
    left = us.ticks[a:b]
    right = us.ticks[b:c]
    whole = us.ticks[a:c]
    assert left.concat(right).equal(whole)


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(uniform_series(min_n=4, max_n=_MAX_N), st.data())
def test_many_cuts_rejoin(backend: str, us: Series, data):
    us = _to_backend(us, backend)
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [us.t_start_ticks, *data.draw(_cuts(us, k)), us.t_end_ticks]
    parts = [us.ticks[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(us)


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(uniform_series(max_n=_MAX_N))
def test_full_domain_slice_is_equal(backend: str, us: Series):
    us = _to_backend(us, backend)
    assert us.ticks[us.t_start_ticks : us.t_end_ticks].equal(us)


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(multi_mic_uniform_series(min_n=4, max_n=_MAX_N), st.data())
def test_multi_mic_slice_concat_identity(backend: str, us: Series, data):
    us = _to_backend(us, backend)
    a, b, c = data.draw(_cuts(us, 3))
    left = us.ticks[a:b]
    right = us.ticks[b:c]
    whole = us.ticks[a:c]
    assert (
        left.slice["mic", np.array([0, 1])]
        .concat(right.slice["mic", np.array([0, 1])])
        .equal(whole.slice["mic", np.array([0, 1])])
    )
    assert left.concat(right).equal(whole)


# ---------------------------------------------------------------------------
# Round-trip type identity: a numpy payload stays numpy, torch stays torch,
# jax stays jax — the additive-only guarantee behind the 0.2.0 bump.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(uniform_series(min_n=4, max_n=_MAX_N), st.data())
def test_backend_preserved_through_ops(backend: str, us: Series, data):
    us = _to_backend(us, backend)
    assert _is_native(us.data, backend)
    a, b, c = data.draw(_cuts(us, 3))
    assert _is_native(us.ticks[a:b].data, backend)  # slice
    assert _is_native(us.ticks[a:b].concat(us.ticks[b:c]).data, backend)  # concat
    assert _is_native(us.shift(1).data, backend)  # shift (anchor-only, O(1))


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(multi_mic_uniform_series(max_n=_MAX_N))
def test_take_dim_preserves_backend(backend: str, us: Series):
    us = _to_backend(us, backend)
    dropped = us.slice["mic", 0]  # int selector drops the mic axis
    assert _is_native(dropped.data, backend)
    assert dropped.dims == ("time",)


# ---------------------------------------------------------------------------
# interpolate always returns float64 numpy, whatever the source backend.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@settings(deadline=None, max_examples=_EX)
@given(uniform_series(min_n=2, max_n=_MAX_N))
def test_interpolate_returns_numpy_f64(backend: str, us: Series):
    us = _to_backend(us, backend)
    query = np.linspace(us.t_start, us.t_end, 7)
    out = us.interpolate(query)
    assert is_numpy_array(out)
    assert out.dtype == np.float64
    assert out.shape == (7,)


# ---------------------------------------------------------------------------
# Cross-backend arrays are never .equal() (distinct namespaces).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "numpy"])
@settings(deadline=None, max_examples=_EX)
@given(uniform_series(max_n=_MAX_N))
def test_cross_backend_is_not_equal(backend: str, us: Series):
    assert not _to_backend(us, backend).equal(us)
