"""Hypothesis strategies for ``tflib`` property tests.

Ports the reference strategies from harmonic-noise-suppression's
``tests/utils/data/strategies.py`` to the ``tflib`` ``Series``/``Frame`` API.
Series are built through the public factory functions (``uniform``/
``events``/``spans``) with exact int ticks, so slice/concat invariants can be
checked exactly (``.equal()``, never ``allclose``).

All time is int64 ticks at ``TICKS_PER_SECOND``.  Tick anchors are drawn both
at small magnitudes and at Unix-epoch magnitude (~1.6e18 ticks, i.e. dates
around the present) to exercise the exact-integer-arithmetic paths that only
show their seams at large offsets.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from tflib._ticks import TICKS_PER_SECOND
from tflib.series import Series, events, spans, uniform

# ---- tick anchors ---------------------------------------------------------
# Small magnitudes (up to ~3 hours in ns) and Unix-epoch magnitude (~2020s).

_small_ticks = st.integers(min_value=0, max_value=10_000_000_000_000)
_large_ticks = st.integers(min_value=1_600_000_000_000_000_000, max_value=1_800_000_000_000_000_000)
tick_anchors = st.one_of(_small_ticks, _large_ticks)

# Integer sample rates plausible for audio / telemetry.
sample_rates = st.sampled_from([8000, 16000, 44100, 48000])

# Domain durations, in ticks (up to 10 s) -- used for stamp/span series.
durations_ticks = st.integers(min_value=1, max_value=10_000_000_000)

# Small dimension sizes for a named "mic"/"rotor"-style leading axis.
dim_sizes = st.integers(min_value=2, max_value=6)


# ---- in-domain cut points -------------------------------------------------


@st.composite
def cut_points_ticks(draw, t_start: int, t_end: int, k: int) -> list[int]:
    """``k`` distinct, ordered int64 tick cut points strictly inside
    ``[t_start, t_end)``."""
    if t_end <= t_start:
        return [t_start] * k
    span = t_end - t_start
    if span < k:
        # Not enough room for k distinct points; caller must tolerate
        # degenerate (identical) cut points.
        return [t_start] * k
    pts = sorted(
        draw(
            st.lists(
                st.integers(min_value=t_start, max_value=t_end - 1),
                min_size=k,
                max_size=k,
                unique=True,
            )
        )
    )
    return pts


# ---- sorted stamp arrays with values --------------------------------------


@st.composite
def sorted_stamps_with_values(
    draw, t_start: int, dur_ticks: int, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """``m`` distinct sorted absolute-tick stamps inside
    ``[t_start, t_start + dur_ticks)``, plus a same-length float payload."""
    if m == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )
    raw = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=dur_ticks - 1),
                min_size=m,
                max_size=m,
                unique=True,
            )
        )
    )
    stamps = np.array(raw, dtype=np.int64) + t_start
    values = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=m,
                max_size=m,
            )
        ),
        dtype=np.float64,
    )
    return stamps, values


# ---- span sets (possibly overlapping) -------------------------------------


@st.composite
def span_bounds(draw, t_start: int, dur_ticks: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """``m`` absolute-tick ``(start, end)`` pairs inside
    ``[t_start, t_start + dur_ticks)``, sorted by start; may overlap."""
    if m == 0 or dur_ticks < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    raw_starts = draw(
        st.lists(
            st.integers(min_value=0, max_value=dur_ticks - 2),
            min_size=m,
            max_size=m,
        )
    )
    pairs = []
    for s in raw_starts:
        e = draw(st.integers(min_value=s + 1, max_value=dur_ticks - 1))
        pairs.append((s, e))
    pairs.sort(key=lambda p: p[0])
    starts = np.array([p[0] for p in pairs], dtype=np.int64) + t_start
    ends = np.array([p[1] for p in pairs], dtype=np.int64) + t_start
    return starts, ends


# ---- multi-dim payloads ("mic"/"rotor"-style leading dim) -----------------


@st.composite
def multi_dim_payload(draw, n_dim: int, n_time: int) -> np.ndarray:
    """A ``(n_dim, n_time)`` float64 payload for a named leading dim
    (e.g. ``"mic"`` or ``"rotor"``) paired with a trailing ``"time"`` dim."""
    flat = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_dim * n_time,
            max_size=n_dim * n_time,
        )
    )
    return np.asarray(flat, dtype=np.float64).reshape(n_dim, n_time)


# ---- Series-returning composites ------------------------------------------


@st.composite
def uniform_series(draw, min_n: int = 1, max_n: int = 64) -> Series:
    """Mono ``uniform()`` series (dims default to ``("time",)``)."""
    sr = draw(sample_rates)
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
    return uniform(data, sr=sr, t_start=t0)


@st.composite
def multi_mic_uniform_series(
    draw, min_n: int = 4, max_n: int = 64, min_mic: int = 2, max_mic: int = 5
) -> Series:
    """``uniform()`` series with a named ``("mic", "time")`` payload."""
    sr = draw(sample_rates)
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    n_mic = draw(st.integers(min_value=min_mic, max_value=max_mic))
    t0 = draw(tick_anchors)
    data = draw(multi_dim_payload(n_mic, n))
    return uniform(data, sr=sr, dims=("mic", "time"), t_start=t0)


@st.composite
def event_series(draw, max_m: int = 32) -> Series:
    """Mono ``events()`` series (dims default to ``("time",)``)."""
    t0 = draw(tick_anchors)
    dur_ticks = draw(durations_ticks)
    # A domain of D ticks holds at most D distinct stamps.
    m = draw(st.integers(min_value=0, max_value=min(max_m, dur_ticks)))
    stamps, values = draw(sorted_stamps_with_values(t0, dur_ticks, m))
    return events(stamps, values, t_start=t0, t_end=t0 + dur_ticks)


@st.composite
def multi_rotor_event_series(
    draw, max_m: int = 16, min_rotor: int = 2, max_rotor: int = 4
) -> Series:
    """``events()`` series with a named ``("rotor", "time")`` payload."""
    t0 = draw(tick_anchors)
    dur_ticks = draw(durations_ticks)
    m = draw(st.integers(min_value=1, max_value=min(max_m, dur_ticks)))
    n_rotor = draw(st.integers(min_value=min_rotor, max_value=max_rotor))
    stamps, _ = draw(sorted_stamps_with_values(t0, dur_ticks, m))
    values = draw(multi_dim_payload(n_rotor, m))
    return events(stamps, values, dims=("rotor", "time"), t_start=t0, t_end=t0 + dur_ticks)


@st.composite
def span_series(draw, max_m: int = 8) -> Series:
    """Mono ``spans()`` series (possibly-overlapping spans; index-only, no
    values -- matching the "vad"-style usage in ``DESIGN.md``)."""
    t0 = draw(tick_anchors)
    dur_ticks = draw(st.integers(min_value=100, max_value=10_000_000_000))
    m = draw(st.integers(min_value=0, max_value=max_m))
    starts, ends = draw(span_bounds(t0, dur_ticks, m))
    return spans(starts, ends, t_start=t0, t_end=t0 + dur_ticks)


@st.composite
def span_series_with_values(draw, max_m: int = 8) -> Series:
    """Mono ``spans()`` series carrying a float payload, one entry per span."""
    t0 = draw(tick_anchors)
    dur_ticks = draw(st.integers(min_value=100, max_value=10_000_000_000))
    m = draw(st.integers(min_value=1, max_value=max_m))
    starts, ends = draw(span_bounds(t0, dur_ticks, m))
    values = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=starts.shape[0],
                max_size=starts.shape[0],
            )
        ),
        dtype=np.float64,
    )
    return spans(starts, ends, values, t_start=t0, t_end=t0 + dur_ticks)


__all__ = [
    "TICKS_PER_SECOND",
    "cut_points_ticks",
    "dim_sizes",
    "durations_ticks",
    "event_series",
    "multi_dim_payload",
    "multi_mic_uniform_series",
    "multi_rotor_event_series",
    "sample_rates",
    "sorted_stamps_with_values",
    "span_bounds",
    "span_series",
    "span_series_with_values",
    "tick_anchors",
    "uniform_series",
]
