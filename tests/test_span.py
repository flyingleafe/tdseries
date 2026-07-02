"""Invariants for ``Series`` built with ``tdseries.spans()`` (``SpanIndex``) --
half-open intervals with identity ids that split on a cut and re-merge on
concat."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdseries.errors import DomainError
from tdseries.indexes import SpanIndex
from tdseries.series import Series, spans

from .strategies import cut_points_ticks, span_series, span_series_with_values


def _span_index(s: Series) -> SpanIndex:
    ti = s.tindex
    assert isinstance(ti, SpanIndex)
    return ti


@st.composite
def _cuts(draw, ss: Series, k: int) -> list[int]:
    return draw(cut_points_ticks(ss.t_start_ticks, ss.t_end_ticks, k))


# ---------------------------------------------------------------------------
# Slice/concat identity
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(span_series())
def test_full_domain_slice_is_equal(ss: Series):
    assert ss.ticks[ss.t_start_ticks : ss.t_end_ticks].equal(ss)


@settings(deadline=None, max_examples=200)
@given(span_series(), st.data())
def test_slice_concat_identity(ss: Series, data):
    a, b, c = data.draw(_cuts(ss, 3))
    joined = ss.ticks[a:b].concat(ss.ticks[b:c])
    assert joined.equal(ss.ticks[a:c])


@settings(deadline=None, max_examples=100)
@given(span_series(), st.data())
def test_many_cuts_rejoin(ss: Series, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [ss.t_start_ticks, *data.draw(_cuts(ss, k)), ss.t_end_ticks]
    parts = [ss.ticks[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(ss)


@settings(deadline=None, max_examples=100)
@given(span_series_with_values(), st.data())
def test_slice_concat_identity_with_values(ss: Series, data):
    a, b, c = data.draw(_cuts(ss, 3))
    joined = ss.ticks[a:b].concat(ss.ticks[b:c])
    assert joined.equal(ss.ticks[a:c])


# ---------------------------------------------------------------------------
# A span straddling a cut splits into two rows sharing an id, and re-merges
# on concat
# ---------------------------------------------------------------------------


def test_straddling_span_splits_and_remerges():
    # One span [0.2 s, 0.8 s) in a 1 s domain; cut at 0.5 s.
    dur = 1_000_000_000
    ss = spans(
        np.array([200_000_000], dtype=np.int64),
        np.array([800_000_000], dtype=np.int64),
        ids=np.array([42], dtype=np.int64),
        t_start=0,
        t_end=dur,
    )
    left = ss.ticks[0:500_000_000]
    right = ss.ticks[500_000_000:dur]

    li = _span_index(left)
    ri = _span_index(right)
    # Both halves hold one row, clipped to the cut, sharing id 42.
    assert li.n == 1
    assert ri.n == 1
    assert int(li.abs_starts_ticks[0]) == 200_000_000
    assert int(li.abs_ends_ticks[0]) == 500_000_000
    assert int(ri.abs_starts_ticks[0]) == 500_000_000
    assert int(ri.abs_ends_ticks[0]) == 800_000_000
    assert int(li.ids[0]) == int(ri.ids[0]) == 42

    rejoined = left.concat(right)
    ji = _span_index(rejoined)
    assert ji.n == 1
    assert int(ji.abs_starts_ticks[0]) == 200_000_000
    assert int(ji.abs_ends_ticks[0]) == 800_000_000
    assert int(ji.ids[0]) == 42
    assert rejoined.equal(ss)


def test_unrelated_spans_meeting_at_seam_are_not_merged():
    # Two distinct spans touching at t = 0.5 s -- different ids, no merge.
    a = spans(
        np.array([200_000_000], dtype=np.int64),
        np.array([500_000_000], dtype=np.int64),
        ids=np.array([1], dtype=np.int64),
        t_start=0,
        t_end=500_000_000,
    )
    b = spans(
        np.array([500_000_000], dtype=np.int64),
        np.array([800_000_000], dtype=np.int64),
        ids=np.array([2], dtype=np.int64),
        t_start=500_000_000,
        t_end=1_000_000_000,
    )
    joined = a.concat(b)
    ji = _span_index(joined)
    assert ji.n == 2
    assert list(ji.abs_starts_ticks) == [200_000_000, 500_000_000]
    assert list(ji.abs_ends_ticks) == [500_000_000, 800_000_000]
    assert list(ji.ids) == [1, 2]


# ---------------------------------------------------------------------------
# Values follow the kept rows
# ---------------------------------------------------------------------------


def test_values_follow_kept_rows_on_slice():
    # Three spans; the middle one is cut away entirely by the window.
    ss = spans(
        np.array([0, 400, 800], dtype=np.int64),
        np.array([100, 500, 900], dtype=np.int64),
        np.array([10.0, 20.0, 30.0]),
        ids=np.array([1, 2, 3], dtype=np.int64),
        t_start=0,
        t_end=1000,
    )
    sliced = ss.ticks[600:1000]  # only the third span survives
    idx = _span_index(sliced)
    assert idx.n == 1
    assert int(idx.ids[0]) == 3
    assert list(np.asarray(sliced.data)) == [30.0]


def test_values_on_straddle_are_duplicated_then_merged():
    ss = spans(
        np.array([200], dtype=np.int64),
        np.array([800], dtype=np.int64),
        np.array([7.5]),
        ids=np.array([9], dtype=np.int64),
        t_start=0,
        t_end=1000,
    )
    left = ss.ticks[0:500]
    right = ss.ticks[500:1000]
    # The straddled span's value appears on both halves.
    assert list(np.asarray(left.data)) == [7.5]
    assert list(np.asarray(right.data)) == [7.5]
    rejoined = left.concat(right)
    # Merged back into a single row with a single value.
    assert list(np.asarray(rejoined.data)) == [7.5]
    assert rejoined.equal(ss)


# ---------------------------------------------------------------------------
# Overlapping spans survive the algebra
# ---------------------------------------------------------------------------


def test_overlapping_spans_slice_concat_identity():
    ss = spans(
        np.array([0, 100, 150], dtype=np.int64),
        np.array([500, 700, 300], dtype=np.int64),
        ids=np.array([1, 2, 3], dtype=np.int64),
        t_start=0,
        t_end=1000,
    )
    joined = ss.ticks[0:250].concat(ss.ticks[250:1000])
    assert joined.equal(ss)


# ---------------------------------------------------------------------------
# Index-only spans / domain edges / shift
# ---------------------------------------------------------------------------


def test_index_only_spans_have_no_data():
    ss = spans(
        np.array([100], dtype=np.int64), np.array([200], dtype=np.int64), t_start=0, t_end=1000
    )
    assert ss.data is None
    assert ss.dims == ("time",)


def test_fresh_ids_generated_when_not_given():
    ss = spans(
        np.array([100, 300], dtype=np.int64),
        np.array([200, 400], dtype=np.int64),
        t_start=0,
        t_end=1000,
    )
    ids = _span_index(ss).ids
    assert ids.shape == (2,)
    assert int(ids[0]) != int(ids[1])


def test_slice_outside_domain_raises():
    ss = spans(
        np.array([100], dtype=np.int64), np.array([200], dtype=np.int64), t_start=0, t_end=1000
    )
    with pytest.raises(DomainError):
        ss.ticks[-100:500]


def test_shift_roundtrip():
    ss = spans(
        np.array([200_000_000], dtype=np.int64),
        np.array([500_000_000], dtype=np.int64),
        t_start=0,
        t_end=1_000_000_000,
    )
    assert ss.shift(5_000_000_000).shift(-5_000_000_000).equal(ss)


def test_shift_at_unix_magnitude_is_exact():
    t0 = 1_690_000_000_000_000_000
    ss = spans(
        np.array([t0 + 100], dtype=np.int64),
        np.array([t0 + 300], dtype=np.int64),
        t_start=t0,
        t_end=t0 + 1000,
    )
    shifted = ss.shift(1)
    idx = _span_index(shifted)
    assert shifted.t_start_ticks == t0 + 1
    assert int(idx.abs_starts_ticks[0]) == t0 + 101
    assert int(idx.abs_ends_ticks[0]) == t0 + 301
    assert shifted.shift(-1).equal(ss)


def test_spans_float_bounds_are_seconds():
    ss = spans(np.array([0.2, 0.6]), np.array([0.4, 0.9]), t_start=0.0, t_end=1.0)
    idx = _span_index(ss)
    assert ss.t_start_ticks == 0
    assert ss.t_end_ticks == 1_000_000_000
    assert list(idx.abs_starts_ticks) == [200_000_000, 600_000_000]
    assert list(idx.abs_ends_ticks) == [400_000_000, 900_000_000]
