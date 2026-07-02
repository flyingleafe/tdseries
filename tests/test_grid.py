"""Invariants for ``Series`` built with ``tflib.uniform()`` (``GridIndex``) --
the trickiest of the three time-index kinds because of sub-sample cuts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tflib.errors import DomainError, IncompatibleError
from tflib.series import Series, uniform

from .strategies import cut_points_ticks, multi_mic_uniform_series, uniform_series

# ---------------------------------------------------------------------------
# Full-domain slice is a no-op
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(uniform_series())
def test_full_domain_slice_is_equal(us: Series):
    sliced = us.ticks[us.t_start_ticks : us.t_end_ticks]
    assert sliced.equal(us)


@settings(deadline=None, max_examples=100)
@given(uniform_series())
def test_none_bounds_are_domain_bounds(us: Series):
    assert us.ticks[:].equal(us)
    assert us.time[:].equal(us)


# ---------------------------------------------------------------------------
# The big one: slice(a,b) + slice(b,c) == slice(a,c), at arbitrary exact
# ticks -- including sub-sample cuts.
# ---------------------------------------------------------------------------


@st.composite
def _cuts(draw, us: Series, k: int) -> list[int]:
    return draw(cut_points_ticks(us.t_start_ticks, us.t_end_ticks, k))


@settings(deadline=None, max_examples=200)
@given(uniform_series(), st.data())
def test_slice_concat_identity(us: Series, data):
    a, b, c = data.draw(_cuts(us, 3))
    left = us.ticks[a:b]
    right = us.ticks[b:c]
    whole = us.ticks[a:c]
    assert left.concat(right).equal(whole)


@settings(deadline=None, max_examples=100)
@given(uniform_series(min_n=4), st.data())
def test_many_cuts_rejoin(us: Series, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [us.t_start_ticks, *data.draw(_cuts(us, k)), us.t_end_ticks]
    parts = [us.ticks[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(us)


# ---------------------------------------------------------------------------
# Sub-sample cuts: exact-edge cut has no overlap; interior cut duplicates the
# straddled sample on both sides, but slice-concat still rejoins exactly.
# ---------------------------------------------------------------------------


def test_exact_boundary_cut_no_overlap():
    us = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=0)
    left = us.ticks[0:500_000_000]  # exactly on sample 5's left edge (0.5 s)
    right = us.ticks[500_000_000:1_000_000_000]
    assert left.shape == (5,)
    assert right.shape == (5,)
    assert left.concat(right).equal(us)


def test_sub_sample_cut_overlaps_one_sample():
    us = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=0)
    # cut at 0.43 s falls inside sample 4's cell [0.4, 0.5)
    left = us.ticks[0:430_000_000]
    right = us.ticks[430_000_000:1_000_000_000]
    assert left.shape == (5,)  # samples 0..4
    assert right.shape == (6,)  # samples 4..9
    assert left.data[-1] == right.data[0] == 4.0
    assert left.concat(right).equal(us)


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def test_slice_outside_domain_raises():
    us = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=0)
    with pytest.raises(DomainError):
        us.ticks[-500_000_000:500_000_000]
    with pytest.raises(DomainError):
        us.ticks[0:2_000_000_000]


def test_ticks_accessor_rejects_float_bounds():
    us = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=0)
    with pytest.raises(TypeError):
        us.ticks[0.0:0.5]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Concat rejects mismatched sample rates
# ---------------------------------------------------------------------------


def test_concat_mismatched_sample_rate_raises():
    a = uniform(np.arange(10.0), sr=10, t_start=0)
    b = uniform(np.arange(10.0), sr=20, t_start=1_000_000_000)
    with pytest.raises(IncompatibleError):
        a.concat(b)


def test_add_dunder_is_concat():
    a = uniform(np.arange(5.0), sr=10, t_start=0)
    b = uniform(np.arange(5.0), sr=10, t_start=500_000_000)
    assert (a + b).equal(a.concat(b))


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(uniform_series(), st.data())
def test_shift_roundtrip_ticks(us: Series, data):
    dt = data.draw(st.integers(min_value=-10_000_000_000, max_value=10_000_000_000))
    assert us.shift(dt).shift(-dt).equal(us)


def test_shift_roundtrip_seconds():
    us = uniform(np.arange(10.0), sr=10, t_start=3_000_000_000)
    assert us.shift(2.5).shift(-2.5).equal(us)


def test_shift_preserves_data_moves_anchor():
    us = uniform(np.arange(10.0), sr=10, t_start=0)
    shifted = us.shift(5_000_000_000)
    assert shifted.t_start_ticks == 5_000_000_000
    assert shifted.t_end_ticks == 6_000_000_000
    assert np.array_equal(shifted.data, us.data)


def test_shift_at_unix_magnitude_anchor_is_exact():
    t0 = 1_700_000_000_000_000_000
    us = uniform(np.arange(10.0), sr=10, t_start=t0)
    shifted = us.shift(1)  # +1 tick
    assert shifted.t_start_ticks == t0 + 1
    assert shifted.shift(-1).equal(us)


# ---------------------------------------------------------------------------
# Multi-dim ("mic", "time") payloads co-slice correctly
# ---------------------------------------------------------------------------


def test_multi_mic_slice_keeps_mic_axis():
    data = np.random.RandomState(0).randn(4, 100).astype(np.float64)
    us = uniform(data, sr=100, dims=("mic", "time"), t_start=10_000_000_000)
    sliced = us.ticks[10_370_000_000:10_830_000_000]
    assert sliced.shape[0] == 4  # mic axis untouched
    assert sliced.dims == ("mic", "time")


@settings(deadline=None, max_examples=100)
@given(multi_mic_uniform_series())
def test_multi_mic_full_domain_slice_is_equal(us: Series):
    assert us.ticks[us.t_start_ticks : us.t_end_ticks].equal(us)


@settings(deadline=None, max_examples=100)
@given(multi_mic_uniform_series(min_n=4), st.data())
def test_multi_mic_slice_concat_identity(us: Series, data):
    a, b, c = data.draw(_cuts(us, 3))
    left = us.ticks[a:b]
    right = us.ticks[b:c]
    whole = us.ticks[a:c]
    assert left.dims == right.dims == whole.dims == ("mic", "time")
    assert left.shape[0] == right.shape[0] == whole.shape[0]
    assert left.concat(right).equal(whole)


def test_multi_mic_concat_rejoins_channel_data():
    data = np.arange(4 * 10, dtype=np.float64).reshape(4, 10)
    us = uniform(data, sr=10, dims=("mic", "time"), t_start=0)
    a = us.ticks[0:500_000_000]
    b = us.ticks[500_000_000:1_000_000_000]
    assert a.concat(b).equal(us)


# ---------------------------------------------------------------------------
# Factory default dims
# ---------------------------------------------------------------------------


def test_uniform_default_dims_mono():
    us = uniform(np.arange(5.0), sr=10)
    assert us.dims == ("time",)


def test_uniform_default_dims_multichannel():
    us = uniform(np.zeros((2, 5)), sr=10)
    assert us.dims == (None, "time")
