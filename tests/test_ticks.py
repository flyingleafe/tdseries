"""Tests for ``tdseries._ticks`` -- the seconds<->int64-tick conversion
boundary, and its two scalar coercion rules (``to_ticks`` vs
``seconds_to_ticks_exact``)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdseries._ticks import (
    TICKS_PER_SECOND,
    seconds_to_ticks_exact,
    secs_array_to_ticks,
    secs_to_ticks,
    ticks_array_to_secs,
    ticks_to_secs,
    to_ticks,
)

from .strategies import tick_anchors

# ---- scalar secs_to_ticks / ticks_to_secs ---------------------------------


def test_secs_to_ticks_whole_seconds():
    assert secs_to_ticks(0.0) == 0
    assert secs_to_ticks(1.0) == TICKS_PER_SECOND
    assert secs_to_ticks(2.0) == 2 * TICKS_PER_SECOND


def test_secs_to_ticks_rounds_to_nearest():
    assert secs_to_ticks(0.5) == 500_000_000
    # round-half-to-even for an exact .5 ns boundary is not reachable at this
    # magnitude; just check ordinary nearest-tick rounding.
    assert secs_to_ticks(0.123456789) == 123_456_789


def test_secs_to_ticks_roundtrip_small_magnitude():
    for x in (0.0, 1.0, 0.123456789, 3600.5):
        assert ticks_to_secs(secs_to_ticks(x)) == pytest.approx(x, abs=1e-9)


@settings(deadline=None, max_examples=100)
@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_secs_to_ticks_roundtrip_property(x: float):
    assert ticks_to_secs(secs_to_ticks(x)) == pytest.approx(x, abs=1e-6)


# ---- to_ticks coercion: float=seconds (quantised), int=ticks (identity) ---


def test_to_ticks_int_is_identity():
    assert to_ticks(42) == 42
    assert to_ticks(0) == 0
    assert to_ticks(TICKS_PER_SECOND) == TICKS_PER_SECOND
    assert to_ticks(-5) == -5


def test_to_ticks_float_is_secs_to_ticks():
    assert to_ticks(1.0) == secs_to_ticks(1.0) == TICKS_PER_SECOND
    assert to_ticks(0.5) == secs_to_ticks(0.5) == 500_000_000


@settings(deadline=None, max_examples=100)
@given(tick_anchors)
def test_to_ticks_int_identity_property(t: int):
    """``to_ticks`` never re-quantises an already-exact int tick, even at
    Unix-epoch magnitude (~1.6e18) where float64 would lose precision."""
    assert to_ticks(t) == t


# ---- seconds_to_ticks_exact: int=WHOLE SECONDS (exact), float=quantised --


def test_seconds_to_ticks_exact_int_is_whole_seconds():
    assert seconds_to_ticks_exact(1) == TICKS_PER_SECOND
    assert seconds_to_ticks_exact(0) == 0
    assert seconds_to_ticks_exact(5) == 5 * TICKS_PER_SECOND
    assert seconds_to_ticks_exact(-2) == -2 * TICKS_PER_SECOND


def test_seconds_to_ticks_exact_float_is_secs_to_ticks():
    assert seconds_to_ticks_exact(0.5) == secs_to_ticks(0.5) == 500_000_000
    assert seconds_to_ticks_exact(1.5) == secs_to_ticks(1.5)


@settings(deadline=None, max_examples=100)
@given(st.integers(min_value=-10_000_000, max_value=10_000_000))
def test_seconds_to_ticks_exact_int_property(n: int):
    """Exact multiplication -- no float rounding, unlike ``to_ticks``."""
    assert seconds_to_ticks_exact(n) == n * TICKS_PER_SECOND


# ---- the coercion rules genuinely differ for int input ---------------------


def test_to_ticks_and_seconds_to_ticks_exact_diverge_for_int():
    """The whole point of having two coercions: the same int argument means
    "ticks" to one and "whole seconds" to the other."""
    n = 3
    assert to_ticks(n) == 3
    assert seconds_to_ticks_exact(n) == 3 * TICKS_PER_SECOND
    assert to_ticks(n) != seconds_to_ticks_exact(n)


def test_to_ticks_and_seconds_to_ticks_exact_agree_at_zero():
    assert to_ticks(0) == seconds_to_ticks_exact(0) == 0


def test_to_ticks_and_seconds_to_ticks_exact_agree_for_float():
    """Both coercions treat a float argument identically (seconds, quantised
    once via ``secs_to_ticks``)."""
    for x in (0.0, 0.5, 2.25, -1.75):
        assert to_ticks(x) == seconds_to_ticks_exact(x) == secs_to_ticks(x)


# ---- array conversions ------------------------------------------------------


def test_secs_array_to_ticks_dtype_and_values():
    out = secs_array_to_ticks(np.array([0.0, 0.5, 1.0]))
    assert out.dtype == np.int64
    assert list(out) == [0, 500_000_000, TICKS_PER_SECOND]


def test_ticks_array_to_secs_dtype_and_values():
    out = ticks_array_to_secs(np.array([0, 500_000_000, TICKS_PER_SECOND], dtype=np.int64))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)


def test_array_roundtrip_small_magnitude():
    secs = np.array([0.0, 1.5, 3600.25])
    out = ticks_array_to_secs(secs_array_to_ticks(secs))
    np.testing.assert_allclose(out, secs, atol=1e-9)


def test_secs_array_to_ticks_empty():
    out = secs_array_to_ticks(np.array([], dtype=np.float64))
    assert out.dtype == np.int64
    assert out.shape == (0,)
