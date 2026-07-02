"""Invariants for ``Series`` built with ``tflib.uniform()`` (``GridIndex``) --
the trickiest of the three time-index kinds because of sub-sample cuts."""

from __future__ import annotations

import operator
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tflib.errors import DomainError, IncompatibleError
from tflib.indexes import GridIndex
from tflib.series import Series, uniform

from .strategies import (
    cut_points_ticks,
    multi_mic_uniform_series,
    tick_anchors,
    uniform_series,
)

# Framed-transform rates: a base audio rate divided by an STFT hop gives an
# exact fraction (e.g. 44100 / 256) that only stays exact as a rational.
_framed_hops = st.sampled_from([160, 256, 441, 512])
_base_srs = st.sampled_from([8000, 16000, 44100, 48000])


@st.composite
def rational_uniform_series(draw, min_n: int = 1, max_n: int = 64) -> Series:
    """Mono series on a ``GridIndex`` whose rate is an exact fraction
    ``Fraction(base_sr, hop)`` -- the framed-transform (STFT) case."""
    sr = draw(_base_srs)
    hop = draw(_framed_hops)
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    t0 = draw(tick_anchors)
    data = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=np.float64,
    )
    idx = GridIndex.create(Fraction(sr, hop), n, t_start=t0)
    return Series(data, ("time",), {"time": idx})


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


def test_plus_operator_is_not_concat():
    # ``+`` is reserved for future aligned element-wise ops; concatenation is
    # the named ``.concat`` method only.
    a = uniform(np.arange(5.0), sr=10, t_start=0)
    b = uniform(np.arange(5.0), sr=10, t_start=500_000_000)
    with pytest.raises(TypeError):
        operator.add(a, b)


# ---------------------------------------------------------------------------
# Rational (framed-transform) sample rates
# ---------------------------------------------------------------------------


def test_integral_float_sr_is_accepted():
    us = uniform(np.arange(5.0), sr=44100.0)
    ti = us.tindex
    assert isinstance(ti, GridIndex)
    assert ti.sr_num == 44100
    assert ti.sr_den == 1


def test_non_integral_float_sr_raises():
    with pytest.raises(ValueError):
        uniform(np.arange(5.0), sr=44100 / 256)  # 172.265625, not integral
    with pytest.raises(ValueError):
        GridIndex.create(172.265625, 5)


def test_fraction_and_tuple_rates_normalize_equally():
    a = GridIndex.create(Fraction(44100, 256), 10)
    b = GridIndex.create((44100, 256), 10)
    c = GridIndex.create((88200, 512), 10)  # same reduced fraction
    # gcd(44100, 256) == 4, so the stored normalized pair is (11025, 64).
    assert (a.sr_num, a.sr_den) == (11025, 64)
    assert a.equal(b)
    assert a.equal(c)
    assert a.rate == Fraction(44100, 256)


@settings(deadline=None, max_examples=200)
@given(rational_uniform_series(), st.data())
def test_rational_slice_concat_identity(us: Series, data):
    a, b, c = data.draw(_cuts(us, 3))
    left = us.ticks[a:b]
    right = us.ticks[b:c]
    whole = us.ticks[a:c]
    assert left.concat(right).equal(whole)


@settings(deadline=None, max_examples=100)
@given(rational_uniform_series(min_n=4), st.data())
def test_rational_many_cuts_rejoin(us: Series, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [us.t_start_ticks, *data.draw(_cuts(us, k)), us.t_end_ticks]
    parts = [us.ticks[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(us)


# ---------------------------------------------------------------------------
# Framed-rate roundtrip: dividing by / multiplying by an STFT hop is stable
# under exact-fraction normalisation (no float drift), even after slice/shift.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
@pytest.mark.parametrize("hop", [160, 256, 441, 512])
def test_framed_rate_normalization_stable(sr: int, hop: int):
    t0 = 1_700_000_000_000_000_000
    base = GridIndex.create(sr, 100, t_start=t0)
    framed_rate = Fraction(base.sr_num, base.sr_den * hop)
    assert framed_rate == Fraction(sr, hop)
    # Multiplying the framed rate back by the hop must reduce exactly to the
    # original audio rate; a grid rebuilt with it .equal()s the original.
    mapped_back = framed_rate * hop
    rebuilt = GridIndex.create(mapped_back, 100, t_start=t0)
    assert rebuilt.equal(base)


@settings(deadline=None, max_examples=100)
@given(rational_uniform_series(min_n=2), st.data())
def test_framed_rate_roundtrip_after_ops(us: Series, data):
    hop = data.draw(_framed_hops)
    a, b = data.draw(_cuts(us, 2))
    dt = data.draw(st.integers(min_value=-1_000_000_000_000, max_value=1_000_000_000_000))
    ti = us.ticks[a:b].shift(dt).tindex
    assert isinstance(ti, GridIndex)
    framed_rate = Fraction(ti.sr_num, ti.sr_den * hop)
    mapped_back = framed_rate * hop
    assert mapped_back == Fraction(ti.sr_num, ti.sr_den)
    rebuilt = GridIndex(
        sr_num=mapped_back.numerator,
        size=ti.size,
        sr_den=mapped_back.denominator,
        t_start_ticks=ti.t_start_ticks,
        dur_ticks=ti.dur_ticks,
        phase=ti.phase,
    )
    assert rebuilt.equal(ti)


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(uniform_series(), st.data())
def test_shift_roundtrip_ticks(us: Series, data):
    dt = data.draw(st.integers(min_value=-10_000_000_000, max_value=10_000_000_000))
    assert us.shift(dt).shift(-dt).equal(us)


@settings(deadline=None, max_examples=100)
@given(rational_uniform_series(), st.data())
def test_rational_shift_roundtrip_ticks(us: Series, data):
    """Shift roundtrips exactly for fractional rates, including at the
    Unix-epoch-magnitude anchors drawn by the strategy."""
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
