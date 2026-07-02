"""Invariants for ``Series`` built with ``tflib.events()`` (``StampIndex``) --
exact int64 tick storage of sorted point events."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tflib.errors import DomainError, IncompatibleError
from tflib.indexes import StampIndex
from tflib.series import Series, events

from .strategies import (
    cut_points_ticks,
    event_series,
    multi_rotor_event_series,
    tick_anchors,
)


def _stamp_index(s: Series) -> StampIndex:
    ti = s.tindex
    assert isinstance(ti, StampIndex)
    return ti


@st.composite
def _cuts(draw, es: Series, k: int) -> list[int]:
    return draw(cut_points_ticks(es.t_start_ticks, es.t_end_ticks, k))


# ---------------------------------------------------------------------------
# Slice/concat identity
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(event_series())
def test_full_domain_slice_is_equal(es: Series):
    assert es.ticks[es.t_start_ticks : es.t_end_ticks].equal(es)


@settings(deadline=None, max_examples=200)
@given(event_series(), st.data())
def test_slice_concat_identity(es: Series, data):
    a, b, c = data.draw(_cuts(es, 3))
    joined = es.ticks[a:b].concat(es.ticks[b:c])
    assert joined.equal(es.ticks[a:c])


@settings(deadline=None, max_examples=100)
@given(event_series(), st.data())
def test_many_cuts_rejoin(es: Series, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [es.t_start_ticks, *data.draw(_cuts(es, k)), es.t_end_ticks]
    parts = [es.ticks[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(es)


# ---------------------------------------------------------------------------
# Half-open boundary routing: an event exactly at the cut lands right
# ---------------------------------------------------------------------------


def test_event_at_cut_goes_right():
    t0 = 10_000_000_000  # 10 s
    dur = 1_000_000_000  # 1 s
    stamps = np.array([t0, t0 + 500_000_000, t0 + 900_000_000], dtype=np.int64)
    vals = np.array([10.0, 20.0, 30.0])
    es = events(stamps, vals, t_start=t0, t_end=t0 + dur)
    t_cut = t0 + 500_000_000  # exactly at the second event
    left = es.ticks[t0:t_cut]
    right = es.ticks[t_cut : t0 + dur]
    assert list(_stamp_index(left).abs_stamps_ticks) == [t0]
    assert list(_stamp_index(right).abs_stamps_ticks) == [t0 + 500_000_000, t0 + 900_000_000]
    assert list(left.data) == [10.0]
    assert list(right.data) == [20.0, 30.0]
    assert left.concat(right).equal(es)


def test_event_at_domain_start_is_kept():
    es = events(np.array([0], dtype=np.int64), np.array([1.0]), t_start=0, t_end=100)
    half = es.ticks[0:50]
    assert half.shape == (1,)


# ---------------------------------------------------------------------------
# Values arrays stay aligned with stamps
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(event_series(), st.data())
def test_values_stay_aligned_with_stamps(es: Series, data):
    a, b = data.draw(_cuts(es, 2))
    sliced = es.ticks[a:b]
    idx = _stamp_index(sliced)
    # One value per stamp, in stamp order.
    assert sliced.shape == (idx.n,)
    # Each surviving (stamp, value) pair exists in the original with the
    # same pairing.
    orig_idx = _stamp_index(es)
    orig_pairs = dict(
        zip(
            (int(t) for t in orig_idx.abs_stamps_ticks),
            (float(v) for v in np.asarray(es.data)),
            strict=True,
        )
    )
    for t, v in zip(idx.abs_stamps_ticks, np.asarray(sliced.data), strict=True):
        assert orig_pairs[int(t)] == float(v)


def test_multi_rotor_values_slice_time_last_axis():
    # 4-rotor RPS stored (rotor, time): slicing must cut the time (last) axis
    # and keep all 4 rotors.
    m = 10
    stamps = np.arange(m, dtype=np.int64) * 100_000_000  # 0, 0.1 s, ... 0.9 s
    vals = np.stack(
        [np.arange(m), np.arange(m) + 100, np.arange(m) + 200, np.arange(m) + 300]
    ).astype(np.float64)  # (4, M)
    es = events(stamps, vals, dims=("rotor", "time"), t_start=0, t_end=1_000_000_000)
    sl = es.ticks[300_000_000:700_000_000]  # events at 0.3, 0.4, 0.5, 0.6
    assert sl.shape == (4, 4)
    np.testing.assert_array_equal(np.asarray(sl.data)[:, 0], [3, 103, 203, 303])
    rejoined = es.ticks[0:300_000_000].concat(es.ticks[300_000_000:1_000_000_000])
    assert rejoined.equal(es)


@settings(deadline=None, max_examples=100)
@given(multi_rotor_event_series(), st.data())
def test_multi_rotor_slice_concat_identity(es: Series, data):
    a, b, c = data.draw(_cuts(es, 3))
    joined = es.ticks[a:b].concat(es.ticks[b:c])
    assert joined.equal(es.ticks[a:c])


# ---------------------------------------------------------------------------
# Shift exactness at Unix-magnitude anchors
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(tick_anchors, st.integers(min_value=-(10**15), max_value=10**15))
def test_shift_roundtrip_at_any_anchor(t0: int, dt: int):
    stamps = np.array([t0 + 1, t0 + 500], dtype=np.int64)
    es = events(stamps, np.array([1.0, 2.0]), t_start=t0, t_end=t0 + 1000)
    assert es.shift(dt).shift(-dt).equal(es)


def test_shift_one_tick_at_unix_magnitude_is_exact():
    t0 = 1_650_000_000_000_000_000
    stamps = np.array([t0, t0 + 123_456_789], dtype=np.int64)
    es = events(stamps, np.array([1.0, 2.0]), t_start=t0, t_end=t0 + 1_000_000_000)
    shifted = es.shift(1)
    assert shifted.t_start_ticks == t0 + 1
    assert list(_stamp_index(shifted).abs_stamps_ticks) == [t0 + 1, t0 + 123_456_790]
    # Relative storage untouched -- shift is O(1) anchor move.
    assert np.array_equal(_stamp_index(shifted).stamps, _stamp_index(es).stamps)


def test_shift_float_seconds_at_unix_magnitude():
    t0 = 1_700_000_000_000_000_000
    es = events(np.array([t0], dtype=np.int64), np.array([5.0]), t_start=t0, t_end=t0 + 100)
    shifted = es.shift(2.5)  # float = seconds, quantised once
    assert shifted.t_start_ticks == t0 + 2_500_000_000
    assert shifted.shift(-2.5).equal(es)


# ---------------------------------------------------------------------------
# Index-only series (values=None)
# ---------------------------------------------------------------------------


def test_index_only_events_have_no_data():
    es = events(np.array([100, 200], dtype=np.int64), t_start=0, t_end=1000)
    assert es.data is None
    assert es.dims == ("time",)
    assert es.shape == (2,)


def test_index_only_slice_concat_identity():
    es = events(np.array([100, 200, 700], dtype=np.int64), t_start=0, t_end=1000)
    joined = es.ticks[0:200].concat(es.ticks[200:1000])
    assert joined.equal(es)


def test_concat_index_only_with_data_raises():
    a = events(np.array([100], dtype=np.int64), np.array([1.0]), t_start=0, t_end=500)
    b = events(np.array([600], dtype=np.int64), t_start=500, t_end=1000)
    with pytest.raises(IncompatibleError):
        a.concat(b)


# ---------------------------------------------------------------------------
# Empty / domain edge cases
# ---------------------------------------------------------------------------


def test_empty_series_slices_to_empty():
    es = events(np.array([], dtype=np.int64), t_start=0, t_end=1_000_000_000)
    half = es.ticks[0:500_000_000]
    assert half.shape == (0,)
    assert half.t_end_ticks == 500_000_000


def test_slice_outside_domain_raises():
    es = events(
        np.array([100_000_000], dtype=np.int64), np.array([1.0]), t_start=0, t_end=1_000_000_000
    )
    with pytest.raises(DomainError):
        es.ticks[-100_000_000:500_000_000]


def test_events_float_timestamps_are_seconds():
    es = events(np.array([0.5, 0.8]), np.array([10.0, 20.0]), t_start=0.0, t_end=1.0)
    assert es.t_start_ticks == 0
    assert es.t_end_ticks == 1_000_000_000
    assert list(_stamp_index(es).abs_stamps_ticks) == [500_000_000, 800_000_000]
