"""``Frame`` — the immutable pytree node keyed by name.

A ``Frame`` holds a dict of named entries.  Two kinds:

* **temporal** — a :class:`~tdseries.series.Series` with a time dim, or a nested
  ``Frame``.  These share the frame's single absolute anchor: the frame owns
  ``t_start_ticks`` + ``dur_ticks`` (the hull of its temporal children) and
  stores each temporal child **relative** to that anchor, so ``shift`` is O(1)
  and reading a child back (``frame[key]``) re-anchors it to absolute time.
* **invariant** — an atemporal ``Series`` (e.g. mic positions) or any other
  Python scalar/object.  Raw numpy/torch tensors are auto-wrapped via
  :func:`~tdseries.series.wrap` so an entry is always a ``Series``, ``Frame``, or
  plain value.

Time algebra (``slice`` / ``concat`` / ``shift``) mirrors the old
``TimeFrame``; dim algebra (``slice`` / ``sel``) recurses over every ``Series``
leaf.  A tree-wide dim check runs at construction: each non-time dim's
``RangeIndex`` occurrences must agree in size, and a dim may not mix
``RangeIndex`` and ``LabelIndex`` occurrences.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._array import is_tensor
from ._ticks import seconds_to_ticks_exact, ticks_to_secs, to_ticks
from .errors import DimensionError, DomainError, IncompatibleError
from .indexes import LabelIndex
from .series import Series, wrap

# =========================================================================
# Entry helpers — classify / re-anchor / compare
# =========================================================================


def _wrap_entry(value: Any) -> Any:
    """Normalise an incoming entry: raw arrays become atemporal ``Series``;
    ``Series`` / ``Frame`` / plain values pass through unchanged."""
    if isinstance(value, (Series, Frame)):
        return value
    if is_tensor(value):
        return wrap(value)
    return value


def _is_temporal(value: Any) -> bool:
    """True for a ``Series`` carrying a time dim, or a nested ``Frame`` that
    (recursively) contains one.  A frame holding only invariant entries — a
    pure metadata bundle — is itself invariant: it survives time slicing and
    is never re-anchored."""
    if isinstance(value, Frame):
        return any(_is_temporal(v) for v in value.entries.values())
    return isinstance(value, Series) and value.has_time


def _abs_start(value: Any) -> int:
    return value.t_start_ticks


def _abs_end(value: Any) -> int:
    return value.t_end_ticks


def _shift_entry(value: Any, dt_ticks: int) -> Any:
    """Shift a temporal entry (``Series`` or ``Frame``) by ``dt`` ticks."""
    return value.shift(dt_ticks)


def _slice_entry(value: Any, a_ticks: int, b_ticks: int) -> Any:
    return value.slice_ticks(a_ticks, b_ticks)


def _concat_entry(left: Any, right: Any) -> Any:
    return left.concat(right)


def _invariant_equal(a: Any, b: Any) -> bool:
    """Equality for two invariant entries.  ``Series`` compare via ``.equal``;
    other values via ``==`` (raw arrays are wrapped to ``Series`` at
    construction, so ``==`` never lands on a bare ndarray)."""
    if isinstance(a, Series) or isinstance(b, Series):
        return isinstance(a, Series) and isinstance(b, Series) and a.equal(b)
    return bool(a == b)


# =========================================================================
# Frame
# =========================================================================


@dataclass(frozen=True, eq=False, init=False)
class Frame:
    """Immutable, dict-like container of named temporal and invariant entries.

    The mapping passed to the constructor is interpreted as **absolute**;
    temporal children are re-based to the frame anchor and validated to fit
    the hull ``[t_start, t_end)`` (inferred from the children when not given).
    """

    entries: dict[str, Any] = field(default_factory=dict)
    t_start_ticks: int = 0
    dur_ticks: int = 0

    # ---- construction ---------------------------------------------------
    def __init__(
        self,
        entries: Mapping[str, Any] | None = None,
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
    ) -> None:
        norm = {k: _wrap_entry(v) for k, v in (entries or {}).items()}
        temporal = [v for v in norm.values() if _is_temporal(v)]
        if temporal:
            hull_start = min(_abs_start(v) for v in temporal)
            hull_end = max(_abs_end(v) for v in temporal)
        else:
            hull_start = 0
            hull_end = 0
        t0 = hull_start if t_start is None else to_ticks(t_start)
        te = hull_end if t_end is None else to_ticks(t_end)
        if temporal:
            if t0 > hull_start:
                raise DomainError(f"frame t_start ({t0}) is after hull start ({hull_start})")
            if te < hull_end:
                raise DomainError(f"frame t_end ({te}) is before hull end ({hull_end})")
        if te < t0:
            raise ValueError(f"frame duration < 0 (t_start={t0} > t_end={te})")

        local = {k: (_shift_entry(v, -t0) if _is_temporal(v) else v) for k, v in norm.items()}
        object.__setattr__(self, "entries", local)
        object.__setattr__(self, "t_start_ticks", t0)
        object.__setattr__(self, "dur_ticks", te - t0)
        _validate_dims(self)

    @classmethod
    def _from_local(
        cls, local_entries: Mapping[str, Any], t_start_ticks: int, dur_ticks: int
    ) -> Frame:
        """Build from entries already stored frame-relative; bypasses
        re-basing and validation (internal fast path)."""
        self = object.__new__(cls)
        object.__setattr__(self, "entries", dict(local_entries))
        object.__setattr__(self, "t_start_ticks", int(t_start_ticks))
        object.__setattr__(self, "dur_ticks", int(dur_ticks))
        return self

    def _abs(self) -> dict[str, Any]:
        """Entries re-anchored to absolute time (temporal ones shifted)."""
        if self.t_start_ticks == 0:
            return dict(self.entries)
        out: dict[str, Any] = {}
        for k, v in self.entries.items():
            out[k] = _shift_entry(v, self.t_start_ticks) if _is_temporal(v) else v
        return out

    # ---- domain ---------------------------------------------------------
    @property
    def t_start(self) -> float:
        return ticks_to_secs(self.t_start_ticks)

    @property
    def t_end(self) -> float:
        return ticks_to_secs(self.t_start_ticks + self.dur_ticks)

    @property
    def t_end_ticks(self) -> int:
        return self.t_start_ticks + self.dur_ticks

    @property
    def duration(self) -> float:
        return ticks_to_secs(self.dur_ticks)

    @property
    def duration_ticks(self) -> int:
        return self.dur_ticks

    # ---- dict-like ------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        v = self.entries[key]
        if self.t_start_ticks != 0 and _is_temporal(v):
            return _shift_entry(v, self.t_start_ticks)
        return v

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> Iterable[str]:
        return self.entries.keys()

    def values(self) -> Iterable[Any]:
        return self._abs().values()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._abs().items()

    # ---- column ops -----------------------------------------------------
    def select(self, keys: Iterable[str]) -> Frame:
        keys = list(keys)
        missing = [k for k in keys if k not in self.entries]
        if missing:
            raise KeyError(f"missing entries: {missing}")
        return Frame._from_local(
            {k: self.entries[k] for k in keys}, self.t_start_ticks, self.dur_ticks
        )

    def drop(self, keys: Iterable[str]) -> Frame:
        drop_set = set(keys)
        return Frame._from_local(
            {k: v for k, v in self.entries.items() if k not in drop_set},
            self.t_start_ticks,
            self.dur_ticks,
        )

    def _set_entry(self, name: str, value: Any, *, expand: bool) -> Frame:
        """Return a new frame with ``name`` set to ``value`` (an absolute
        entry).  With ``expand=True`` the hull grows to fit a temporal value;
        with ``expand=False`` the declared domain is preserved and a temporal
        value that extends past it raises ``DomainError``."""
        value = _wrap_entry(value)
        abs_entries = {**self._abs(), name: value}
        if expand and _is_temporal(value):
            new_t0 = min(self.t_start_ticks, _abs_start(value))
            new_te = max(self.t_end_ticks, _abs_end(value))
        else:
            new_t0 = self.t_start_ticks
            new_te = self.t_end_ticks
        return Frame(abs_entries, t_start=new_t0, t_end=new_te)

    def with_entry(self, name: str, value: Any, *, expand: bool = True) -> Frame:
        """Return a new frame with ``name`` set to ``value``.  By default the
        hull expands if a temporal value extends past it; ``expand=False``
        keeps the declared domain and raises ``DomainError`` if the value
        does not fit."""
        return self._set_entry(name, value, expand=expand)

    def map_entry(self, name: str, fn: Callable[[Any], Any], *, expand: bool = False) -> Frame:
        """Apply ``fn`` to the entry ``name`` — the value ``frame[name]``
        returns, a temporal child re-anchored to absolute time — and store
        the result back under the same name.  By default the declared domain
        is preserved and a temporal result must fit inside it (``DomainError``
        otherwise); ``expand=True`` gives the domain-expanding put-semantics
        of ``with_entry``."""
        return self._set_entry(name, fn(self[name]), expand=expand)

    def merge(self, other: Frame, overwrite: bool = False) -> Frame:
        """Column-wise union of two frames; result hull is the union.  Key
        collisions raise unless ``overwrite``."""
        if not overwrite:
            collisions = set(self.entries) & set(other.entries)
            if collisions:
                raise IncompatibleError(
                    f"key collisions on merge: {sorted(collisions)} (pass overwrite=True)"
                )
        abs_entries = {**self._abs(), **other._abs()}
        new_t0 = min(self.t_start_ticks, other.t_start_ticks)
        new_te = max(self.t_end_ticks, other.t_end_ticks)
        return Frame(abs_entries, t_start=new_t0, t_end=new_te)

    # ---- time ops -------------------------------------------------------
    @property
    def time(self) -> _FrameTimeSlicer:
        return _FrameTimeSlicer(self)

    @property
    def ticks(self) -> _FrameTicksSlicer:
        return _FrameTicksSlicer(self)

    def shift(self, dt: float | int) -> Frame:
        d = to_ticks(dt)
        if d == 0:
            return self
        return Frame._from_local(self.entries, self.t_start_ticks + d, self.dur_ticks)

    def slice_ticks(self, a_ticks: int, b_ticks: int) -> Frame:
        """Restrict to absolute ``[a, b)`` ticks; the window must lie inside
        the frame domain.  Each temporal child is clipped to its overlap and
        dropped if disjoint; invariant entries pass through."""
        t0, t1 = self.t_start_ticks, self.t_end_ticks
        if a_ticks < t0 or b_ticks > t1 or a_ticks > b_ticks:
            raise DomainError(f"slice({a_ticks}, {b_ticks}) outside [{t0}, {t1}] (ticks)")
        new_entries: dict[str, Any] = {}
        for k, v in self._abs().items():
            if _is_temporal(v):
                a_eff = max(a_ticks, _abs_start(v))
                b_eff = min(b_ticks, _abs_end(v))
                if a_eff <= b_eff:
                    new_entries[k] = _slice_entry(v, a_eff, b_eff)
            else:
                new_entries[k] = v
        return Frame(new_entries, t_start=a_ticks, t_end=b_ticks)

    def concat(self, other: Frame) -> Frame:
        """Glue ``other`` along time so its domain begins at
        ``self.t_end_ticks``.  Keys union; temporal children concat, invariant
        entries must agree, one-sided entries carry over (temporal shifted)."""
        delta = self.dur_ticks
        self_abs = self._abs()
        other_abs = other._abs()
        new_entries: dict[str, Any] = {}
        for name in set(self_abs) | set(other_abs):
            in_self = name in self_abs
            in_other = name in other_abs
            if in_self and in_other:
                a = self_abs[name]
                b = other_abs[name]
                a_temporal = _is_temporal(a)
                b_temporal = _is_temporal(b)
                if a_temporal and b_temporal:
                    new_entries[name] = _concat_entry(
                        a, _shift_entry(b, delta - other.t_start_ticks)
                    )
                elif not a_temporal and not b_temporal:
                    if not _invariant_equal(a, b):
                        raise IncompatibleError(f"invariant entry {name!r} conflicts on concat")
                    new_entries[name] = a
                else:
                    raise IncompatibleError(
                        f"entry {name!r} is temporal on one side, invariant on the other"
                    )
            elif in_self:
                new_entries[name] = self_abs[name]
            else:
                v = other_abs[name]
                new_entries[name] = (
                    _shift_entry(v, delta - other.t_start_ticks) if _is_temporal(v) else v
                )
        return Frame(
            new_entries,
            t_start=self.t_start_ticks,
            t_end=self.t_start_ticks + self.dur_ticks + other.dur_ticks,
        )

    # ---- dim ops (recursive over Series leaves) -------------------------
    @property
    def slice(self) -> _FramePosSlicer:
        return _FramePosSlicer(self)

    @property
    def sel(self) -> _FrameLabelSlicer:
        return _FrameLabelSlicer(self)

    def _map_series(self, fn: Any) -> Frame:
        """Apply ``fn`` to every ``Series`` entry, recursing into nested
        frames; non-Series entries and the time anchor are untouched."""
        new_entries: dict[str, Any] = {}
        for k, v in self.entries.items():
            if isinstance(v, Series):
                new_entries[k] = fn(v)
            elif isinstance(v, Frame):
                new_entries[k] = v._map_series(fn)
            else:
                new_entries[k] = v
        return Frame._from_local(new_entries, self.t_start_ticks, self.dur_ticks)

    def _slice_dim(self, dim: str, sel: Any) -> Frame:
        if dim == "time":
            raise DimensionError("cannot positionally slice 'time'; use .time / .ticks")
        sizes: set[int] = set()
        for _, s in self.leaves():
            if dim in s.dims:
                sizes.add(s.dim_size(dim))
        if not sizes:
            raise DimensionError(f"no leaf carries dim {dim!r}")
        if len(sizes) > 1:
            raise DimensionError(
                f"dim {dim!r} has unequal sizes {sorted(sizes)} across leaves; use .sel"
            )
        return self._map_series(lambda s: s.take_dim(dim, sel) if dim in s.dims else s)

    def _sel_dim(self, dim: str, labels: Any) -> Frame:
        if dim == "time":
            raise DimensionError("cannot label-select 'time'; use .time / .ticks")
        found = False
        for _, s in self.leaves():
            if dim in s.dims:
                found = True
                if not isinstance(s.indexes.get(dim), LabelIndex):
                    raise DimensionError(f"dim {dim!r} lacks a LabelIndex on some leaf")
        if not found:
            raise DimensionError(f"no leaf carries dim {dim!r}")
        return self._map_series(lambda s: s.sel[dim, labels] if dim in s.dims else s)

    # ---- introspection --------------------------------------------------
    def leaves(self, prefix: str = "") -> Iterator[tuple[str, Series]]:
        """Iterate ``(dotted_path, Series)`` over every ``Series`` leaf, with
        each temporal leaf re-anchored to absolute time."""
        for k, v in self._abs().items():
            path = f"{prefix}{k}"
            if isinstance(v, Series):
                yield path, v
            elif isinstance(v, Frame):
                yield from v.leaves(prefix=f"{path}.")

    def map_data(self, fn: Any) -> Frame:
        """Apply ``fn`` to every ``Series`` leaf's data (e.g. numpy to torch),
        preserving structure and anchors."""
        new_entries: dict[str, Any] = {}
        for k, v in self.entries.items():
            if isinstance(v, (Series, Frame)):
                new_entries[k] = v.map_data(fn)
            else:
                new_entries[k] = v
        return Frame._from_local(new_entries, self.t_start_ticks, self.dur_ticks)

    # ---- equality -------------------------------------------------------
    def equal(self, other: object) -> bool:
        """Exact equality: anchors, key set, and every entry (``Series`` /
        ``Frame`` via ``.equal``, other values via ``==``)."""
        if not isinstance(other, Frame):
            return False
        if self.t_start_ticks != other.t_start_ticks or self.dur_ticks != other.dur_ticks:
            return False
        if set(self.entries) != set(other.entries):
            return False
        # Anchors match, so stored (relative) entries can be compared directly.
        for k, va in self.entries.items():
            vb = other.entries[k]
            if isinstance(va, (Series, Frame)):
                if not (type(va) is type(vb) and va.equal(vb)):
                    return False
            elif isinstance(vb, (Series, Frame)) or not _invariant_equal(va, vb):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        return self.equal(other)

    def __hash__(self) -> int:
        return id(self)


# =========================================================================
# Tree-wide dim validation
# =========================================================================


def _validate_dims(frame: Frame) -> None:
    """Check non-time dims across all ``Series`` leaves: ``RangeIndex``
    occurrences of one dim must agree in size, and a dim may not mix
    ``RangeIndex`` and ``LabelIndex`` occurrences."""
    range_sizes: dict[str, int] = {}
    label_dims: set[str] = set()
    range_dims: set[str] = set()
    for _, s in frame.leaves():
        for name in s.dims:
            if name is None or name == "time":
                continue
            if isinstance(s.indexes.get(name), LabelIndex):
                label_dims.add(name)
            else:
                range_dims.add(name)
                size = s.dim_size(name)
                if name in range_sizes and range_sizes[name] != size:
                    raise DimensionError(
                        f"dim {name!r} has conflicting RangeIndex sizes "
                        f"({range_sizes[name]} vs {size})"
                    )
                range_sizes[name] = size
    mixed = label_dims & range_dims
    if mixed:
        raise DimensionError(
            f"dim(s) {sorted(mixed)} mix RangeIndex and LabelIndex occurrences; "
            "give every occurrence a LabelIndex"
        )


# =========================================================================
# Accessors
# =========================================================================


class _FrameTimeSlicer:
    """``frame.time[a:b]`` — bounds are float seconds (``int`` = whole
    seconds); ``None`` = frame domain edge; step forbidden."""

    def __init__(self, frame: Frame) -> None:
        self._f = frame

    def __getitem__(self, key: object) -> Frame:
        if not isinstance(key, slice):
            raise TypeError("use .time[a:b] with a slice")
        if key.step is not None:
            raise ValueError("slice step is not supported on .time")
        f = self._f
        a = f.t_start_ticks if key.start is None else seconds_to_ticks_exact(key.start)
        b = f.t_end_ticks if key.stop is None else seconds_to_ticks_exact(key.stop)
        return f.slice_ticks(a, b)


class _FrameTicksSlicer:
    """``frame.ticks[a:b]`` — bounds are exact int64 ticks; ``None`` = frame
    domain edge; step forbidden."""

    def __init__(self, frame: Frame) -> None:
        self._f = frame

    def __getitem__(self, key: object) -> Frame:
        if not isinstance(key, slice):
            raise TypeError("use .ticks[a:b] with a slice")
        if key.step is not None:
            raise ValueError("slice step is not supported on .ticks")
        if key.start is not None and not isinstance(key.start, (int, np.integer)):
            raise TypeError("ticks bounds must be int; use .time for seconds")
        if key.stop is not None and not isinstance(key.stop, (int, np.integer)):
            raise TypeError("ticks bounds must be int; use .time for seconds")
        f = self._f
        a = f.t_start_ticks if key.start is None else int(key.start)
        b = f.t_end_ticks if key.stop is None else int(key.stop)
        return f.slice_ticks(a, b)


class _FramePosSlicer:
    """``frame.slice[dim, sel]`` — positional selection along a non-time dim,
    recursively over every leaf that carries it."""

    def __init__(self, frame: Frame) -> None:
        self._f = frame

    def __getitem__(self, key: tuple[str, Any]) -> Frame:
        dim, sel = key
        return self._f._slice_dim(dim, sel)


class _FrameLabelSlicer:
    """``frame.sel[dim, labels]`` — label selection; every occurrence of the
    dim must carry a ``LabelIndex``."""

    def __init__(self, frame: Frame) -> None:
        self._f = frame

    def __getitem__(self, key: tuple[str, Any]) -> Frame:
        dim, labels = key
        return self._f._sel_dim(dim, labels)
