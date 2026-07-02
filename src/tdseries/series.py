"""``Series`` — the leaf tensor with named, indexed dimensions.

A ``Series`` wraps one array (numpy or torch) together with a ``dims`` tuple
(one name, or ``None``, per axis) and an ``indexes`` mapping from dim name to
its :class:`~tdseries.indexes.DimIndex` / :class:`~tdseries.indexes.TimeIndex`.  Time
lives on the reserved ``"time"`` dim and carries a ``TimeIndex``; all time
algebra (slice / concat / shift) is delegated to that index, which returns
positional selections that this module applies to the data axis.

Storage is anchor-relative inside the ``TimeIndex`` (see ``indexes.py``), so
``shift`` is O(1) and tick arithmetic is exact int64 throughout.  The public
seconds accessors quantise once, at the boundary, via ``_ticks``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._array import array_equal, take, to_numpy_f64
from ._array import concat as concat_arrays
from ._ticks import (
    seconds_to_ticks_exact,
    ticks_array_to_secs,
    ticks_to_secs,
    to_ticks,
)
from .errors import DimensionError, IncompatibleError
from .indexes import (
    DimIndex,
    GridIndex,
    LabelIndex,
    RangeIndex,
    SampleRate,
    SpanIndex,
    StampIndex,
    TimeIndex,
    normalize_rate,
)

# A positional selection along one non-time axis (as accepted by ``.slice``):
# ``int`` (drops the dim) | ``slice`` | int/bool array | list of ints.
PosSel = int | slice | list | np.ndarray


def _dim_index_equal(a: DimIndex | TimeIndex, b: DimIndex | TimeIndex) -> bool:
    """Structural equality across the index union, narrowing each side to the
    same family before delegating to its own ``.equal``."""
    if isinstance(a, TimeIndex):
        return isinstance(b, TimeIndex) and a.equal(b)
    return isinstance(b, DimIndex) and a.equal(b)


# =========================================================================
# Accessor helpers
# =========================================================================
#
# Each ``__getitem__`` on a Series accessor is a thin translator into one of
# the Series methods; they exist only to give the ``s.time[a:b]`` /
# ``s.slice[dim, sel]`` surface syntax.


class _TimeSlicer:
    """``s.time[a:b]`` — bounds are float **seconds** (``int`` = whole
    seconds); ``None`` = domain edge; step forbidden."""

    def __init__(self, series: Series) -> None:
        self._s = series

    def __getitem__(self, key: object) -> Series:
        if not isinstance(key, slice):
            raise TypeError("use .time[a:b] with a slice")
        if key.step is not None:
            raise ValueError("slice step is not supported on .time")
        ti = self._s.tindex
        a = ti.t_start_ticks if key.start is None else seconds_to_ticks_exact(key.start)
        b = ti.t_end_ticks if key.stop is None else seconds_to_ticks_exact(key.stop)
        return self._s.slice_ticks(a, b)


class _TicksSlicer:
    """``s.ticks[a:b]`` — bounds are exact int64 **ticks** (``TypeError`` on
    float); ``None`` = domain edge; step forbidden."""

    def __init__(self, series: Series) -> None:
        self._s = series

    def __getitem__(self, key: object) -> Series:
        if not isinstance(key, slice):
            raise TypeError("use .ticks[a:b] with a slice")
        if key.step is not None:
            raise ValueError("slice step is not supported on .ticks")
        if key.start is not None and not isinstance(key.start, (int, np.integer)):
            raise TypeError("ticks bounds must be int; use .time for seconds")
        if key.stop is not None and not isinstance(key.stop, (int, np.integer)):
            raise TypeError("ticks bounds must be int; use .time for seconds")
        ti = self._s.tindex
        a = ti.t_start_ticks if key.start is None else int(key.start)
        b = ti.t_end_ticks if key.stop is None else int(key.stop)
        return self._s.slice_ticks(a, b)


class _PosSlicer:
    """``s.slice[dim, sel]`` — positional selection along a named non-time
    dim; ``int`` drops the dim (numpy semantics)."""

    def __init__(self, series: Series) -> None:
        self._s = series

    def __getitem__(self, key: tuple[str, PosSel]) -> Series:
        dim, sel = key
        if dim == "time":
            raise DimensionError("cannot positionally slice 'time'; use .time / .ticks")
        if dim not in self._s.dims:
            raise DimensionError(f"unknown dim {dim!r}")
        return self._s.take_dim(dim, sel)


class _LabelSlicer:
    """``s.sel[dim, label_or_list]`` — label selection; requires a
    ``LabelIndex`` on the dim."""

    def __init__(self, series: Series) -> None:
        self._s = series

    def __getitem__(self, key: tuple[str, Any]) -> Series:
        dim, want = key
        if dim == "time":
            raise DimensionError("cannot label-select 'time'; use .time / .ticks")
        if dim not in self._s.dims:
            raise DimensionError(f"unknown dim {dim!r}")
        idx = self._s.indexes.get(dim)
        if not isinstance(idx, LabelIndex):
            raise DimensionError(f"dim {dim!r} has no LabelIndex; use .slice for positional")
        return self._s.take_dim(dim, idx.locate(want))


# =========================================================================
# Series
# =========================================================================


@dataclass(frozen=True, eq=False)
class Series:
    """One array plus named, indexed dimensions.

    ``data`` may be a numpy array, a torch tensor, or ``None`` (an index-only
    series, which must have ``dims == ("time",)`` and a ``TimeIndex``).
    ``dims`` gives one name — or ``None`` for an anonymous axis — per data
    axis.  ``indexes`` maps a *subset* of the named dims to their index; the
    reserved ``"time"`` dim always carries a ``TimeIndex``, non-time named dims
    default to ``RangeIndex(size)`` when absent.
    """

    data: Any
    dims: tuple[str | None, ...]
    indexes: Mapping[str, DimIndex | TimeIndex] = field(default_factory=dict)

    # ---- validation -----------------------------------------------------
    def __post_init__(self) -> None:
        if self.indexes is None:
            object.__setattr__(self, "indexes", {})
        elif not isinstance(self.indexes, dict):
            object.__setattr__(self, "indexes", dict(self.indexes))
        object.__setattr__(self, "dims", tuple(self.dims))

        dims = self.dims
        indexes = self.indexes
        named = [d for d in dims if d is not None]
        if len(set(named)) != len(named):
            raise DimensionError(f"duplicate named dims in {dims}")
        for key in indexes:
            if key not in named:
                raise DimensionError(f"index key {key!r} is not a named dim of {dims}")

        if self.data is None:
            if dims != ("time",):
                raise ValueError("index-only Series (data=None) must have dims == ('time',)")
            if not isinstance(indexes.get("time"), TimeIndex):
                raise DimensionError("'time' dim requires a TimeIndex")
            return

        if len(dims) != self.data.ndim:
            raise DimensionError(f"len(dims)={len(dims)} != data.ndim={self.data.ndim}")
        for axis, name in enumerate(dims):
            if name is None:
                continue
            size = int(self.data.shape[axis])
            if name == "time":
                ti = indexes.get("time")
                if not isinstance(ti, TimeIndex):
                    raise DimensionError("'time' dim requires a TimeIndex")
                if ti.n != size:
                    raise DimensionError(f"time index n={ti.n} != axis size {size}")
            else:
                di = indexes.get(name)
                if di is None:
                    continue
                if isinstance(di, TimeIndex):
                    raise DimensionError(f"TimeIndex may only sit on 'time', not {name!r}")
                if di.n != size:
                    raise DimensionError(f"index for {name!r} has n={di.n} != axis size {size}")

    # ---- shape / dims ---------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        if self.data is not None:
            return tuple(self.data.shape)
        return (self.tindex.n,)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def has_time(self) -> bool:
        return "time" in self.dims

    @property
    def time_axis(self) -> int | None:
        return self.dims.index("time") if "time" in self.dims else None

    @property
    def tindex(self) -> TimeIndex:
        ti = self.indexes.get("time")
        if not isinstance(ti, TimeIndex):
            raise ValueError("Series is atemporal (no 'time' dim)")
        return ti

    def _time_axis(self) -> int:
        return self.dims.index("time")

    # ---- domain (seconds + exact ticks) ---------------------------------
    @property
    def t_start(self) -> float:
        return self.tindex.t_start

    @property
    def t_end(self) -> float:
        return self.tindex.t_end

    @property
    def duration(self) -> float:
        return self.tindex.duration

    @property
    def t_start_ticks(self) -> int:
        return self.tindex.t_start_ticks

    @property
    def t_end_ticks(self) -> int:
        return self.tindex.t_end_ticks

    @property
    def duration_ticks(self) -> int:
        return self.tindex.dur_ticks

    def dim_index(self, name: str) -> DimIndex | TimeIndex:
        """The explicit index for ``name``, or the default ``RangeIndex`` for
        a non-time named dim that carries none."""
        if name not in self.dims:
            raise DimensionError(f"unknown dim {name!r}")
        if name in self.indexes:
            return self.indexes[name]
        return RangeIndex(self.dim_size(name))

    def dim_size(self, name: str) -> int:
        if name not in self.dims:
            raise DimensionError(f"unknown dim {name!r}")
        return self.shape[self.dims.index(name)]

    # ---- accessors ------------------------------------------------------
    @property
    def time(self) -> _TimeSlicer:
        return _TimeSlicer(self)

    @property
    def ticks(self) -> _TicksSlicer:
        return _TicksSlicer(self)

    @property
    def slice(self) -> _PosSlicer:
        return _PosSlicer(self)

    @property
    def sel(self) -> _LabelSlicer:
        return _LabelSlicer(self)

    # ---- time ops -------------------------------------------------------
    def slice_ticks(self, a_ticks: int, b_ticks: int) -> Series:
        """Restrict the time dim to absolute ``[a, b)`` ticks."""
        new_index, possel = self.tindex.slice(a_ticks, b_ticks)
        new_data = self.data
        if new_data is not None and possel is not None:
            new_data = take(new_data, self._time_axis(), possel)
        return Series(new_data, self.dims, {**self.indexes, "time": new_index})

    def shift(self, dt: float | int) -> Series:
        """Move the whole timeline by ``dt`` (float seconds or int ticks).
        O(1); atemporal series raise ``ValueError``."""
        return Series(
            self.data, self.dims, {**self.indexes, "time": self.tindex.shift(to_ticks(dt))}
        )

    def concat(self, other: object) -> Series:
        """Glue ``other`` (a ``Series``) onto the end of this series along
        the time dim."""
        if not isinstance(other, Series):
            raise IncompatibleError(f"cannot concat Series with {type(other).__name__}")
        if self.dims != other.dims:
            raise IncompatibleError(f"dims differ: {self.dims} vs {other.dims}")
        if (self.data is None) != (other.data is None):
            raise IncompatibleError("one operand is index-only, the other has data")
        for name in self.dims:
            if name is None or name == "time":
                continue
            if not _dim_index_equal(self.dim_index(name), other.dim_index(name)):
                raise IncompatibleError(f"non-time index for dim {name!r} differs")
        new_index, left_sel, right_sel = self.tindex.concat(other.tindex)
        if self.data is None:
            new_data = None
        else:
            axis = self._time_axis()
            left = self.data if left_sel is None else take(self.data, axis, left_sel)
            right = other.data if right_sel is None else take(other.data, axis, right_sel)
            new_data = concat_arrays([left, right], axis)
        return Series(new_data, self.dims, {**self.indexes, "time": new_index})

    # ---- non-time positional / label selection --------------------------
    def take_dim(self, name: str, sel: Any) -> Series:
        """Positional selection along a named non-time dim, co-updating that
        dim's stored index; an ``int`` selector drops the dim."""
        if name == "time":
            raise DimensionError("cannot positionally slice 'time'; use .time / .ticks")
        if name not in self.dims:
            raise DimensionError(f"unknown dim {name!r}")
        axis = self.dims.index(name)
        idx = self.dim_index(name)
        assert isinstance(idx, DimIndex)  # non-time dim
        new_idx = idx.take(sel)
        new_data = take(self.data, axis, sel)
        if new_idx is None:  # int selector drops the dim
            new_dims = self.dims[:axis] + self.dims[axis + 1 :]
            new_indexes = {k: v for k, v in self.indexes.items() if k != name}
        else:
            new_dims = self.dims
            new_indexes = {**self.indexes, name: new_idx}
        return Series(new_data, new_dims, new_indexes)

    # ---- interpolation / resampling -------------------------------------
    def interpolate(self, times: Any, *, kind: str = "linear", fill: str = "clamp") -> np.ndarray:
        """Evaluate the signal at absolute query ``times`` (float seconds or
        int ticks) by linear interpolation over the time axis.

        Works for a ``GridIndex`` (grid = sample edges) and a ``StampIndex``
        (grid = event stamps); a ``SpanIndex`` has no point-wise values and
        raises ``TypeError``.  The time axis of the result keeps its position
        in ``dims``; the output is always float64 numpy.
        """
        if kind != "linear":
            raise ValueError(f"unsupported interpolation kind: {kind!r}")
        ti = self.tindex
        if isinstance(ti, GridIndex):
            grid_t = ti.sample_times()
        elif isinstance(ti, StampIndex):
            grid_t = ti.abs_stamps
        elif isinstance(ti, SpanIndex):
            raise TypeError("SpanIndex has no point-wise values; interpolate is undefined")
        else:
            raise TypeError(f"cannot interpolate over {type(ti).__name__}")
        if self.data is None:
            raise ValueError("cannot interpolate an index-only Series (data is None)")

        times = np.asarray(times)
        t_sec = ticks_array_to_secs(times) if times.dtype.kind == "i" else times.astype(np.float64)
        axis = self._time_axis()
        vals = np.moveaxis(to_numpy_f64(self.data), axis, -1)  # time last
        n = vals.shape[-1]

        if n == 0:
            fill_val = np.nan if fill == "nan" else 0.0
            out = np.full((*vals.shape[:-1], t_sec.shape[0]), fill_val, dtype=np.float64)
            return np.moveaxis(out, -1, axis)

        flat = vals.reshape(-1, n)
        result = np.empty((flat.shape[0], t_sec.shape[0]), dtype=np.float64)
        for c in range(flat.shape[0]):
            result[c, :] = np.interp(t_sec, grid_t, flat[c, :])
        result = result.reshape(*vals.shape[:-1], t_sec.shape[0])

        if fill == "clamp":
            pass  # np.interp already clamps at the edges
        elif fill == "nan":
            mask = (t_sec < grid_t[0]) | (t_sec > grid_t[-1])
            result[..., mask] = np.nan
        elif fill == "error":
            if t_sec[0] < grid_t[0] - 1e-12 or t_sec[-1] > grid_t[-1] + 1e-12:
                raise ValueError(
                    f"query times [{t_sec[0]:.6g}, {t_sec[-1]:.6g}] outside data span "
                    f"[{grid_t[0]:.6g}, {grid_t[-1]:.6g}]"
                )
        else:
            raise ValueError(f"unsupported fill: {fill!r}")
        return np.moveaxis(result, -1, axis)

    def resample(self, new_sr: SampleRate, *, kind: str = "linear") -> Series:
        """Resample onto a fresh phase-0 grid at ``new_sr`` over the same
        declared domain; the result carries a ``GridIndex``.  ``new_sr`` is
        coerced to an exact ``(num, den)`` fraction (see
        :func:`~tdseries.indexes.normalize_rate`)."""
        num, den = normalize_rate(new_sr)
        sr_float = num / den
        ti = self.tindex
        n_new = max(1, round(ticks_to_secs(ti.dur_ticks) * sr_float))
        grid_s = ti.t_start + np.arange(n_new) / sr_float
        vals = self.interpolate(grid_s, kind=kind, fill="clamp")
        new_index = GridIndex.create((num, den), n_new, t_start=ti.t_start_ticks)
        return Series(vals, self.dims, {**self.indexes, "time": new_index})

    # ---- data replacement ----------------------------------------------
    def with_data(self, new_data: Any) -> Series:
        """Replace the data, keeping dims and indexes; shape must match."""
        new_shape = tuple(new_data.shape)
        if new_shape != self.shape:
            raise ValueError(f"with_data shape {new_shape} != current shape {self.shape}")
        return Series(new_data, self.dims, dict(self.indexes))

    def map_data(self, fn: Any) -> Series:
        """``with_data(fn(data))`` sugar; index-only series pass through."""
        if self.data is None:
            return self
        return self.with_data(fn(self.data))

    # ---- equality -------------------------------------------------------
    def equal(self, other: object) -> bool:
        """Exact structural equality: dims, every dim index (defaults
        normalised), and data (``array_equal``)."""
        if not isinstance(other, Series):
            return False
        if self.dims != other.dims:
            return False
        for name in self.dims:
            if name is None:
                continue
            if not _dim_index_equal(self.dim_index(name), other.dim_index(name)):
                return False
        if (self.data is None) != (other.data is None):
            return False
        if self.data is None:
            return True
        return array_equal(self.data, other.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Series):
            return NotImplemented
        return self.equal(other)

    def __hash__(self) -> int:
        return id(self)


# =========================================================================
# Factory functions
# =========================================================================


def _default_dims(ndim: int) -> tuple[str | None, ...]:
    """``(None, ..., "time")`` — anonymous leading axes, time last."""
    return (None,) * (ndim - 1) + ("time",)


def uniform(
    data: Any,
    sr: SampleRate,
    *,
    dims: tuple[str | None, ...] | None = None,
    t_start: float | int = 0.0,
    phase: float = 0.0,
) -> Series:
    """A uniformly-sampled series on a ``GridIndex``.  ``dims`` defaults to
    ``(None, ..., "time")``; ``sr`` is coerced to an exact ``(num, den)``
    fraction (see :func:`~tdseries.indexes.normalize_rate`); ``t_start`` is float
    seconds or int ticks."""
    dims = _default_dims(data.ndim) if dims is None else tuple(dims)
    size = int(data.shape[dims.index("time")])
    idx = GridIndex.create(sr, size, t_start=t_start, phase=phase)
    return Series(data, dims, {"time": idx})


def events(
    timestamps: Any,
    values: Any = None,
    *,
    dims: tuple[str | None, ...] | None = None,
    t_start: float | int | None = None,
    t_end: float | int | None = None,
) -> Series:
    """A point-event series on a ``StampIndex``.  ``values=None`` builds an
    index-only series (``data=None``, ``dims == ("time",)``).  Timestamps are
    float seconds or int ticks."""
    idx = StampIndex.from_stamps(np.asarray(timestamps), t_start=t_start, t_end=t_end)
    if values is None:
        return Series(None, ("time",), {"time": idx})
    dims = _default_dims(values.ndim) if dims is None else tuple(dims)
    return Series(values, dims, {"time": idx})


def spans(
    starts: Any,
    ends: Any,
    values: Any = None,
    *,
    ids: np.ndarray | None = None,
    t_start: float | int | None = None,
    t_end: float | int | None = None,
    dims: tuple[str | None, ...] | None = None,
) -> Series:
    """A half-open interval series on a ``SpanIndex``.  ``values=None`` builds
    an index-only series.  Bounds are float seconds or int ticks."""
    idx = SpanIndex.from_spans(
        np.asarray(starts), np.asarray(ends), ids, t_start=t_start, t_end=t_end
    )
    if values is None:
        return Series(None, ("time",), {"time": idx})
    dims = _default_dims(values.ndim) if dims is None else tuple(dims)
    return Series(values, dims, {"time": idx})


def wrap(
    data: Any,
    dims: tuple[str | None, ...] | None = None,
    indexes: Mapping[str, DimIndex] | None = None,
) -> Series:
    """An atemporal series (no time dim).  ``dims`` defaults to all-anonymous
    ``(None, ..., None)``."""
    resolved: tuple[str | None, ...] = (None,) * data.ndim if dims is None else tuple(dims)
    return Series(data, resolved, dict(indexes or {}))
