"""Dimension indexes — the mapping from tensor positions to a dimension's domain.

A *dimension* is a name shared by axes of several tensors in a frame, plus a
*domain* its positions are mapped into.  Each tensor axis carries an *index*:
a monotone map ``positions 0..n-1 → domain coordinates``.  Slicing is
expressed in domain coordinates and translated per-tensor into positional
selections, which is what lets tensors of *different lengths* share one
dimension coherently.

Two families:

* ``DimIndex`` — ordinary dimensions.

  - ``RangeIndex`` (implicit default): the domain *is* the position range, so
    every tensor sharing the dim must have equal size.
  - ``LabelIndex``: positions carry hashable labels from a shared label
    domain; sizes may differ across tensors, selection by label.

* ``TimeIndex`` — the time dimension.  Domain = int64 ticks at
  ``TICKS_PER_SECOND``.  Every ``TimeIndex`` declares a half-open domain
  ``[t_start_ticks, t_start_ticks + dur_ticks)`` and supports exact
  ``slice`` / ``concat`` / O(1) ``shift``:

  - ``GridIndex``  — uniform sample grid (audio): anchor + count + an exact
    *rational* rate (``sr_num / sr_den``) + sub-sample ``phase``; the
    cut->sample-index map is exact integer arithmetic for every rate.
    Storing the rate as a reduced fraction keeps framed-transform rates such
    as ``44100 / 256`` exact, so wav -> STFT -> iSTFT -> wav roundtrips
    preserve the index; there is no float-rate epsilon fallback.
  - ``StampIndex`` — sorted point events (RPS, IMU): explicit relative
    int64 timestamps.
  - ``SpanIndex``  — half-open intervals (VAD, labels): starts/ends plus
    identity tags so segments split by a cut re-merge exactly on concat.

The invariant every ``TimeIndex`` upholds (property-tested):

    slice(a, b) ⊕ slice(b, c) == slice(a, c)   for t_start ≤ a ≤ b ≤ c ≤ t_end
"""

from __future__ import annotations

import math
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np

from ._ticks import (
    TICKS_PER_SECOND,
    secs_array_to_ticks,
    ticks_array_to_secs,
    ticks_to_secs,
    to_ticks,
)
from .errors import DomainError, IncompatibleError

# A positional selection along one axis: ``None`` means "take all"; an
# ``int`` drops the axis.
PosSel = slice | np.ndarray | int | None

# A sample rate accepted by the ``GridIndex`` constructors: an ``int`` or an
# integral ``float`` (samples per second), an exact ``Fraction``, or a
# ``(numerator, denominator)`` integer tuple.  Non-integral floats are
# rejected -- pass a ``Fraction`` or a ``(num, den)`` tuple instead.
SampleRate = int | float | Fraction | tuple[int, int]


def normalize_rate(sr: SampleRate) -> tuple[int, int]:
    """Coerce a :data:`SampleRate` to a reduced ``(sr_num, sr_den)`` pair with
    both parts ``> 0``.

    A non-integral ``float`` cannot be represented exactly and is rejected;
    the caller should pass a ``Fraction`` (e.g. ``Fraction(44100, 256)``) or a
    ``(num, den)`` tuple to express a fractional rate exactly.
    """
    if isinstance(sr, bool) or not isinstance(sr, (int, float, Fraction, tuple, np.integer)):
        raise TypeError(f"unsupported sample-rate type {type(sr).__name__}")
    if isinstance(sr, tuple):
        num, den = int(sr[0]), int(sr[1])
    elif isinstance(sr, Fraction):
        num, den = sr.numerator, sr.denominator
    elif isinstance(sr, float):
        if not math.isfinite(sr):
            raise ValueError(f"sample rate must be finite, got {sr!r}")
        if not sr.is_integer():
            raise ValueError(
                f"non-integral float sample rate {sr!r} cannot be stored exactly; "
                "pass a Fraction (e.g. Fraction(44100, 256)) or a (num, den) tuple"
            )
        num, den = int(sr), 1
    else:
        num, den = int(sr), 1
    if num <= 0 or den <= 0:
        raise ValueError(f"sample rate must be > 0, got ({num}, {den})")
    g = math.gcd(num, den)
    return num // g, den // g


# =========================================================================
# Ordinary dimensions
# =========================================================================


class DimIndex(ABC):
    """Index of an ordinary (non-time) dimension."""

    @property
    @abstractmethod
    def n(self) -> int: ...

    @abstractmethod
    def take(self, sel: PosSel) -> DimIndex | None:
        """Positional subset.  Returns ``None`` if the dim is dropped
        (integer selector)."""

    @abstractmethod
    def equal(self, other: DimIndex) -> bool: ...


@dataclass(frozen=True)
class RangeIndex(DimIndex):
    """The default index: positions are their own coordinates."""

    size: int

    @property
    def n(self) -> int:
        return self.size

    def take(self, sel: PosSel) -> DimIndex | None:
        if isinstance(sel, (int, np.integer)):
            return None
        return RangeIndex(_sel_length(sel, self.size))

    def equal(self, other: DimIndex) -> bool:
        return isinstance(other, RangeIndex) and other.size == self.size


@dataclass(frozen=True)
class LabelIndex(DimIndex):
    """Positions carry hashable labels from a shared domain (mic names,
    rotor ids, ...).  Tensors sharing a label-indexed dim may have
    *different* sizes; selection by label resolves per-tensor."""

    labels: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")

    @property
    def n(self) -> int:
        return len(self.labels)

    def take(self, sel: PosSel) -> DimIndex | None:
        if isinstance(sel, (int, np.integer)):
            return None
        if isinstance(sel, slice):
            return LabelIndex(self.labels[sel])
        arr = np.asarray(sel)
        if arr.dtype == bool:
            picked = [lbl for lbl, keep in zip(self.labels, arr, strict=True) if keep]
        else:
            picked = [self.labels[int(i)] for i in arr]
        return LabelIndex(tuple(picked))

    def locate(self, want) -> PosSel:
        """Map a label or sequence of labels to a positional selection.
        Labels absent from this index are silently skipped (that position
        simply does not exist in this tensor)."""
        pos = {lbl: i for i, lbl in enumerate(self.labels)}
        if isinstance(want, (list, tuple, np.ndarray)):
            return np.asarray([pos[w] for w in want if w in pos], dtype=np.int64)
        if want not in pos:
            raise KeyError(f"label {want!r} not in index {self.labels!r}")
        return pos[want]

    def equal(self, other: DimIndex) -> bool:
        return isinstance(other, LabelIndex) and other.labels == self.labels


def _sel_length(sel: PosSel, n: int) -> int:
    """Length of a positional selection applied to an axis of size ``n``."""
    if sel is None:
        return n
    if isinstance(sel, slice):
        return len(range(*sel.indices(n)))
    arr = np.asarray(sel)
    if arr.dtype == bool:
        return int(arr.sum())
    return int(arr.shape[0])


# =========================================================================
# Time
# =========================================================================


class TimeIndex(ABC):
    """Index of the time dimension.  Domain = absolute int64 ticks.

    Concrete indexes store their content *relative* to ``t_start_ticks``
    (the single absolute anchor), which is what makes ``shift`` O(1).
    """

    t_start_ticks: int
    dur_ticks: int

    # ---- shared domain accessors -----------------------------------------
    @property
    def t_end_ticks(self) -> int:
        return self.t_start_ticks + self.dur_ticks

    @property
    def t_start(self) -> float:
        return ticks_to_secs(self.t_start_ticks)

    @property
    def t_end(self) -> float:
        return ticks_to_secs(self.t_end_ticks)

    @property
    def duration(self) -> float:
        return ticks_to_secs(self.dur_ticks)

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of positions along the dim (samples / events / spans)."""

    # ---- algebra ----------------------------------------------------------
    def _check_window(self, a: int, b: int) -> tuple[int, int]:
        t0, t1 = self.t_start_ticks, self.t_end_ticks
        if a < t0 or b > t1 or a > b:
            raise DomainError(f"slice({a}, {b}) outside [{t0}, {t1}] (ticks)")
        return max(a, t0), min(b, t1)

    @abstractmethod
    def slice(self, a_ticks: int, b_ticks: int) -> tuple[TimeIndex, PosSel]:
        """Restrict to absolute ``[a, b)`` ticks.  Returns the new index and
        the positional selection to apply to the data axis."""

    @abstractmethod
    def shift(self, dt_ticks: int) -> TimeIndex:
        """Move the whole timeline by ``dt`` ticks.  O(1)."""

    @abstractmethod
    def concat(self, other: TimeIndex) -> tuple[TimeIndex, PosSel, PosSel]:
        """Glue ``other`` so its domain starts at ``self.t_end_ticks``.
        Returns ``(new_index, left_sel, right_sel)`` — the positional
        selections to apply to the two data operands before concatenation."""

    @abstractmethod
    def equal(self, other: TimeIndex) -> bool: ...


# ---- uniform sample grid -------------------------------------------------


@dataclass(frozen=True)
class GridIndex(TimeIndex):
    """Uniform sample grid: position ``k`` occupies the cell starting at
    ``t_start + (phase + k)/sr``.

    No per-sample edge times are stored; every edge is derived from
    ``(t_start_ticks, phase, rate)``.  The rate is an *exact* reduced fraction
    ``sr_num / sr_den`` samples per second (both parts ``> 0``), so
    framed-transform rates such as ``44100 / 256`` stay exact and every
    cut-time -> sample-index map is exact integer ``divmod`` arithmetic --
    there is no float-rate epsilon fallback.  ``phase`` in [-1, 0] is the
    offset of sample 0's left edge from ``t_start`` in sample units -- bounded
    by one sample, so a float64 holds it to ~1e-21 s.
    """

    sr_num: int
    size: int
    sr_den: int = 1
    t_start_ticks: int = 0
    dur_ticks: int = 0
    phase: float = 0.0

    def __post_init__(self) -> None:
        num, den = int(self.sr_num), int(self.sr_den)
        if num <= 0 or den <= 0:
            raise ValueError(f"sr_num ({num}) and sr_den ({den}) must be > 0")
        g = math.gcd(num, den)
        object.__setattr__(self, "sr_num", num // g)
        object.__setattr__(self, "sr_den", den // g)
        if self.phase < -1.0 or self.phase > 0.0:
            raise ValueError(f"phase ({self.phase}) must be in [-1, 0]")
        if self.dur_ticks < 0:
            raise ValueError(f"dur_ticks ({self.dur_ticks}) < 0")

    @classmethod
    def create(
        cls,
        sr: SampleRate,
        size: int,
        *,
        t_start: float | int = 0.0,
        phase: float = 0.0,
        dur: int | None = None,
    ) -> GridIndex:
        """Grid whose declared domain matches the sample grid exactly
        (unless ``dur`` ticks are given explicitly).  ``sr`` is coerced to an
        exact ``(sr_num, sr_den)`` fraction by :func:`normalize_rate`."""
        num, den = normalize_rate(sr)
        t0 = to_ticks(t_start)
        if dur is None:
            # dur = round(size * TICKS_PER_SECOND * sr_den / sr_num), exact
            # integer arithmetic with round-half-away-from-zero (all > 0).
            q, r = divmod(size * TICKS_PER_SECOND * den, num)
            dur = q + (1 if 2 * r >= num else 0)
        return cls(
            sr_num=num, size=int(size), sr_den=den, t_start_ticks=t0, dur_ticks=dur, phase=phase
        )

    @property
    def sr(self) -> float:
        """The sample rate as a float (display / interop)."""
        return self.sr_num / self.sr_den

    @property
    def rate(self) -> Fraction:
        """The exact sample rate as a :class:`fractions.Fraction`."""
        return Fraction(self.sr_num, self.sr_den)

    @property
    def n(self) -> int:
        return self.size

    @property
    def t_first_edge_ticks(self) -> int:
        return self.t_start_ticks + round(self.phase * TICKS_PER_SECOND * self.sr_den / self.sr_num)

    @property
    def t_first_edge(self) -> float:
        return ticks_to_secs(self.t_start_ticks) + self.phase / self.sr

    def sample_times(self) -> np.ndarray:
        """Absolute left-edge time of each sample, float seconds."""
        return self.t_first_edge + np.arange(self.size) / self.sr

    def sample_times_ticks(self) -> np.ndarray:
        """Absolute left-edge time of each sample, int64 ticks (nearest)."""
        edges_s = self.t_first_edge + np.arange(self.size) / self.sr
        return np.rint(edges_s * TICKS_PER_SECOND).astype(np.int64)

    def time_to_position(self, t: float | int) -> int:
        """Position of the sample cell containing time ``t``."""
        delta = to_ticks(t) - self.t_first_edge_ticks
        # floor(delta * sr / tps) = floor(delta * sr_num / (tps * sr_den));
        # integer floor division is exact for a positive denominator.
        return (delta * self.sr_num) // (TICKS_PER_SECOND * self.sr_den)

    # ---- internal: cut-time -> sample-index map (exact for every rate) ----
    def _cut_to_positions(self, ra: int, rb: int) -> tuple[int, int]:
        # k = ra * sr_num / (tps * sr_den) - phase.  With denom = tps * sr_den
        # this is the integer-sr path with ``tps`` replaced by ``denom`` and
        # the sample rate by ``sr_num`` -- exact ``divmod`` throughout.
        denom = TICKS_PER_SECOND * self.sr_den
        num = self.sr_num
        pticks = round(-self.phase * denom)  # in [0, denom], since phase <= 0
        # left (floor): k = floor(ra*num/denom - phase)
        qa, ra_rem = divmod(ra * num, denom)
        ka = qa + (1 if ra_rem + pticks >= denom else 0)
        # right (ceil): k = ceil(rb*num/denom - phase)
        qb, rb_rem = divmod(rb * num, denom)
        if rb_rem + pticks == 0:
            kb = qb
        elif rb_rem + pticks <= denom:
            kb = qb + 1
        else:
            kb = qb + 2
        ka = max(0, ka)
        kb = min(self.size, kb)
        if kb < ka:
            kb = ka
        return ka, kb

    def slice(self, a_ticks: int, b_ticks: int) -> tuple[GridIndex, PosSel]:
        a, b = self._check_window(a_ticks, b_ticks)
        ra = a - self.t_start_ticks
        rb = b - self.t_start_ticks
        ka, kb = self._cut_to_positions(ra, rb)
        # New phase: offset of the new sample 0's left edge from the new
        # t_start, in sample units; consistent with the stored ka.  The
        # fractional part is derived exactly via divmod so only a bounded
        # remainder / denominator becomes float (tiny-error, like before).
        denom = TICKS_PER_SECOND * self.sr_den
        q, rem = divmod(ra * self.sr_num, denom)
        new_phase = self.phase + (ka - q) - rem / denom
        if new_phase > 0.0:
            new_phase -= 1.0
        elif new_phase < -1.0:
            new_phase += 1.0
        idx = GridIndex(
            sr_num=self.sr_num,
            size=kb - ka,
            sr_den=self.sr_den,
            t_start_ticks=a,
            dur_ticks=rb - ra,
            phase=new_phase,
        )
        return idx, slice(ka, kb)

    def shift(self, dt_ticks: int) -> GridIndex:
        if dt_ticks == 0:
            return self
        return replace(self, t_start_ticks=self.t_start_ticks + dt_ticks)

    def concat(self, other: TimeIndex) -> tuple[GridIndex, PosSel, PosSel]:
        if not isinstance(other, GridIndex):
            raise IncompatibleError(f"cannot concat GridIndex with {type(other).__name__}")
        if (self.sr_num, self.sr_den) != (other.sr_num, other.sr_den):
            raise IncompatibleError(f"sample rates differ: {self.sr} vs {other.sr}")
        # Grid offset in sample-index space: ``other`` is glued so its
        # t_start lands at self's t_end.
        denom = TICKS_PER_SECOND * self.sr_den
        sdur = self.dur_ticks
        q, r = divmod(sdur * self.sr_num, denom)
        k_offset = q + r / denom + other.phase - self.phase

        k_int = round(k_offset)
        if abs(k_offset - k_int) > 0.1:
            raise IncompatibleError(
                f"incompatible sample grids: phase offset {k_offset} samples "
                f"(must be integer; nearest is {k_int})"
            )
        if k_int == self.size:
            right_sel: PosSel = None
            right_n = other.size
        elif k_int == self.size - 1:
            # The seam sample exists on both sides — de-duplicate.
            right_sel = slice(1, None)
            right_n = other.size - 1
        else:
            raise IncompatibleError(
                f"incompatible sample grids: integer offset {k_int} "
                f"(expected {self.size} or {self.size - 1})"
            )
        idx = GridIndex(
            sr_num=self.sr_num,
            size=self.size + right_n,
            sr_den=self.sr_den,
            t_start_ticks=self.t_start_ticks,
            dur_ticks=self.dur_ticks + other.dur_ticks,
            phase=self.phase,
        )
        return idx, None, right_sel

    def equal(self, other: TimeIndex) -> bool:
        if not isinstance(other, GridIndex):
            return False
        return (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
            and self.sr_num == other.sr_num
            and self.sr_den == other.sr_den
            and self.size == other.size
            and abs(self.phase - other.phase) <= 1e-12
        )


# ---- point events ----------------------------------------------------------


@dataclass(frozen=True)
class StampIndex(TimeIndex):
    """Sorted point-event timestamps, stored relative to ``t_start_ticks``."""

    stamps: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int64))
    t_start_ticks: int = 0
    dur_ticks: int = 0

    def __post_init__(self) -> None:
        ts = np.asarray(self.stamps, dtype=np.int64)
        object.__setattr__(self, "stamps", ts)
        if ts.ndim != 1:
            raise ValueError("stamps must be 1-D")
        if self.dur_ticks < 0:
            raise ValueError(f"dur_ticks ({self.dur_ticks}) < 0")
        if ts.size:
            if ts[0] < 0:
                raise ValueError("events before t_start (relative < 0)")
            if ts[-1] >= self.dur_ticks:
                raise ValueError(
                    f"events outside domain (relative {ts[-1]} >= dur {self.dur_ticks})"
                )
            if not np.all(np.diff(ts) >= 0):
                raise ValueError("stamps must be sorted ascending")

    @classmethod
    def from_stamps(
        cls,
        timestamps: np.ndarray,
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
    ) -> StampIndex:
        """Build from *absolute* timestamps (float seconds or int ticks);
        domain inferred from the first/last event when not given."""
        ts = np.asarray(timestamps)
        ts_ticks = (
            secs_array_to_ticks(ts) if np.issubdtype(ts.dtype, np.floating) else ts.astype(np.int64)
        )
        t0 = (int(ts_ticks[0]) if ts_ticks.size else 0) if t_start is None else to_ticks(t_start)
        rel = ts_ticks - t0
        if t_end is None:
            dur = int(rel[-1]) + 1 if rel.size else 0
        else:
            dur = to_ticks(t_end) - t0
        return cls(stamps=rel, t_start_ticks=t0, dur_ticks=dur)

    @property
    def n(self) -> int:
        return int(self.stamps.shape[0])

    @property
    def abs_stamps_ticks(self) -> np.ndarray:
        return self.stamps + self.t_start_ticks

    @property
    def abs_stamps(self) -> np.ndarray:
        return ticks_array_to_secs(self.stamps + self.t_start_ticks)

    def slice(self, a_ticks: int, b_ticks: int) -> tuple[StampIndex, PosSel]:
        a, b = self._check_window(a_ticks, b_ticks)
        ra = a - self.t_start_ticks
        rb = b - self.t_start_ticks
        lo = int(np.searchsorted(self.stamps, ra, side="left"))
        hi = int(np.searchsorted(self.stamps, rb, side="left"))  # half-open right
        idx = StampIndex(stamps=self.stamps[lo:hi] - ra, t_start_ticks=a, dur_ticks=rb - ra)
        return idx, slice(lo, hi)

    def shift(self, dt_ticks: int) -> StampIndex:
        if dt_ticks == 0:
            return self
        return replace(self, t_start_ticks=self.t_start_ticks + dt_ticks)

    def concat(self, other: TimeIndex) -> tuple[StampIndex, PosSel, PosSel]:
        if not isinstance(other, StampIndex):
            raise IncompatibleError(f"cannot concat StampIndex with {type(other).__name__}")
        new_stamps = np.concatenate([self.stamps, other.stamps + self.dur_ticks])
        idx = StampIndex(
            stamps=new_stamps,
            t_start_ticks=self.t_start_ticks,
            dur_ticks=self.dur_ticks + other.dur_ticks,
        )
        return idx, None, None

    def equal(self, other: TimeIndex) -> bool:
        if not isinstance(other, StampIndex):
            return False
        return (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
            and bool(np.array_equal(self.stamps, other.stamps))
        )


# ---- half-open spans ---------------------------------------------------------


def _new_ids(count: int) -> np.ndarray:
    """``count`` random 62-bit positive int64 identity tags."""
    return np.array([secrets.randbits(62) for _ in range(count)], dtype=np.int64)


@dataclass(frozen=True)
class SpanIndex(TimeIndex):
    """Sorted half-open intervals ``[start, end)`` relative to
    ``t_start_ticks``; may overlap arbitrarily.

    Each span carries an identity tag: a cut through a span emits two rows
    sharing the tag, and concat at a matching seam re-merges them — the
    algebraic marker that makes ``slice(a,b) ⊕ slice(b,c) == slice(a,c)``
    hold exactly.
    """

    starts: np.ndarray = field(repr=False)
    ends: np.ndarray = field(repr=False)
    ids: np.ndarray = field(repr=False)
    t_start_ticks: int = 0
    dur_ticks: int = 0

    def __post_init__(self) -> None:
        s = np.asarray(self.starts, dtype=np.int64)
        e = np.asarray(self.ends, dtype=np.int64)
        if s.shape != e.shape or s.ndim != 1:
            raise ValueError("starts and ends must be 1-D and same shape")
        if s.size and not np.all(e > s):
            raise ValueError("each span must have end > start")
        if s.size and not np.all(np.diff(s) >= 0):
            raise ValueError("spans must be sorted by start")
        object.__setattr__(self, "starts", s)
        object.__setattr__(self, "ends", e)
        ids = np.asarray(self.ids, dtype=np.int64)
        if ids.shape != (s.shape[0],):
            raise ValueError("ids must be 1-D with length M")
        object.__setattr__(self, "ids", ids)
        if self.dur_ticks < 0:
            raise ValueError(f"dur_ticks ({self.dur_ticks}) < 0")
        if s.size:
            if s[0] < 0:
                raise ValueError("span starts before t_start (relative < 0)")
            if e.max() > self.dur_ticks:
                raise ValueError(
                    f"span ends after t_end (relative {e.max()} > dur {self.dur_ticks})"
                )

    @classmethod
    def from_spans(
        cls,
        starts: np.ndarray,
        ends: np.ndarray,
        ids: np.ndarray | None = None,
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
    ) -> SpanIndex:
        """Build from *absolute* bounds (float seconds or int ticks).
        Fresh identity tags are generated when ``ids`` is not given."""
        s = np.asarray(starts)
        e = np.asarray(ends)
        s_ticks = (
            secs_array_to_ticks(s) if np.issubdtype(s.dtype, np.floating) else s.astype(np.int64)
        )
        e_ticks = (
            secs_array_to_ticks(e) if np.issubdtype(e.dtype, np.floating) else e.astype(np.int64)
        )
        t0 = (int(s_ticks.min()) if s_ticks.size else 0) if t_start is None else to_ticks(t_start)
        rel_s = s_ticks - t0
        rel_e = e_ticks - t0
        if t_end is None:
            dur = int(rel_e.max()) if rel_e.size else 0
        else:
            dur = to_ticks(t_end) - t0
        if ids is None:
            ids = _new_ids(rel_s.shape[0])
        return cls(starts=rel_s, ends=rel_e, ids=ids, t_start_ticks=t0, dur_ticks=dur)

    @property
    def n(self) -> int:
        return int(self.starts.shape[0])

    @property
    def abs_starts_ticks(self) -> np.ndarray:
        return self.starts + self.t_start_ticks

    @property
    def abs_ends_ticks(self) -> np.ndarray:
        return self.ends + self.t_start_ticks

    @property
    def abs_starts(self) -> np.ndarray:
        return ticks_array_to_secs(self.starts + self.t_start_ticks)

    @property
    def abs_ends(self) -> np.ndarray:
        return ticks_array_to_secs(self.ends + self.t_start_ticks)

    def slice(self, a_ticks: int, b_ticks: int) -> tuple[SpanIndex, PosSel]:
        a, b = self._check_window(a_ticks, b_ticks)
        ra = a - self.t_start_ticks
        rb = b - self.t_start_ticks
        s, e = self.starts, self.ends
        keep = (e > ra) & (s < rb)
        pos = np.where(keep)[0]
        ns = np.clip(s[pos], ra, None) - ra
        ne = np.clip(e[pos], None, rb) - ra
        nonzero = ne > ns
        pos = pos[nonzero]
        idx = SpanIndex(
            starts=ns[nonzero],
            ends=ne[nonzero],
            ids=self.ids[pos],
            t_start_ticks=a,
            dur_ticks=rb - ra,
        )
        return idx, pos

    def shift(self, dt_ticks: int) -> SpanIndex:
        if dt_ticks == 0:
            return self
        return replace(self, t_start_ticks=self.t_start_ticks + dt_ticks)

    def concat(self, other: TimeIndex) -> tuple[SpanIndex, PosSel, PosSel]:
        if not isinstance(other, SpanIndex):
            raise IncompatibleError(f"cannot concat SpanIndex with {type(other).__name__}")
        seam = self.dur_ticks
        o_starts = other.starts + seam
        o_ends = other.ends + seam

        # Merge pairs: same id ending at the seam (left) / starting at it (right).
        left_at_seam = {int(self.ids[i]): i for i in np.where(self.ends == seam)[0]}
        right_at_seam = {int(other.ids[j]): j for j in np.where(o_starts == seam)[0]}
        merge_ids = set(left_at_seam) & set(right_at_seam)

        new_left_ends = self.ends.copy()
        right_keep = np.ones(other.n, dtype=bool)
        for mid in merge_ids:
            new_left_ends[left_at_seam[mid]] = o_ends[right_at_seam[mid]]
            right_keep[right_at_seam[mid]] = False

        idx = SpanIndex(
            starts=np.concatenate([self.starts, o_starts[right_keep]]),
            ends=np.concatenate([new_left_ends, o_ends[right_keep]]),
            ids=np.concatenate([self.ids, other.ids[right_keep]]),
            t_start_ticks=self.t_start_ticks,
            dur_ticks=self.dur_ticks + other.dur_ticks,
        )
        # Left rows are kept as-is (only their index bounds changed); the
        # merged right rows are dropped.
        return idx, None, right_keep

    def equal(self, other: TimeIndex) -> bool:
        if not isinstance(other, SpanIndex):
            return False
        return (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
            and bool(np.array_equal(self.starts, other.starts))
            and bool(np.array_equal(self.ends, other.ends))
            and bool(np.array_equal(self.ids, other.ids))
        )
