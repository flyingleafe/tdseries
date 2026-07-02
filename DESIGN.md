# tflib — design contract

Immutable pytree frames of tensors with **named, indexed dimensions**; time is
a first-class dimension indexed in exact int64 **ticks** (`TICKS_PER_SECOND =
1e9`). Successor of `utils.data` (TimeSeries/TimeFrame) in the
harmonic-noise-suppression project; the tick/phase arithmetic is ported from
there verbatim (see that repo's `src/utils/data/` for reference semantics and
`tests/utils/data/` for reference property tests).

## Core model

A **dimension** is a name plus a *domain*. Each tensor axis carrying that name
has an **index**: a monotone map `positions 0..n-1 → domain coordinates`.
Slices are expressed in domain coordinates and translated per-tensor into
positional selections. This is what lets tensors of *different lengths* share
a dimension (audio at 44.1 kHz and RPS events at ~100 Hz both have dim
`"time"`).

Index families (implemented in `tflib/indexes.py`, already written):

| Index | Domain | Sizes across a frame | Selection |
|---|---|---|---|
| `RangeIndex` (implicit default) | positions | must be equal | positional |
| `LabelIndex` | hashable labels | may differ | by label (`.sel`) or positional |
| `GridIndex` | ticks (uniform grid + sub-sample phase) | may differ | `.time` / `.ticks` |
| `StampIndex` | ticks (sorted point events) | may differ | `.time` / `.ticks` |
| `SpanIndex` | ticks (half-open intervals + identity ids) | may differ | `.time` / `.ticks` |

The dim name `"time"` is reserved: it must carry a `TimeIndex`
(`GridIndex | StampIndex | SpanIndex`), and a `TimeIndex` may only sit on
`"time"`. At most one `"time"` dim per tensor.

`TimeIndex` methods return `(new_index, positional_selection)` so the caller
(Series) applies the selection to its data axis. All three uphold, exactly:

    slice(a, b) ⊕ slice(b, c) == slice(a, c)   for t_start ≤ a ≤ b ≤ c ≤ t_end

`shift` is O(1) everywhere (only the scalar anchor moves; stored content is
anchor-relative).

## `Series` — the leaf (`tflib/series.py`)

Frozen dataclass, `eq=False`:

```python
Series(
    data,          # np.ndarray | torch.Tensor | None  (None ⇒ index-only, dims must be ("time",))
    dims,          # tuple[str | None, ...], len == data.ndim; None = anonymous axis
    indexes={},    # Mapping[str, DimIndex | TimeIndex]; keys ⊆ named dims
)
```

Validation: dims unique among named; `"time"` ⇒ `indexes["time"]` is a
`TimeIndex` with `n ==` axis size; non-time named dims may carry a
`RangeIndex`/`LabelIndex` with matching `n`; anonymous (`None`) axes carry no
index. Use `tflib._array.is_tensor/take/concat/array_equal/to_numpy_f64` for
all data manipulation (they dispatch over the numpy and torch backends).

Properties: `shape`, `ndim`, `has_time`, `time_axis`, `tindex` (raises
`ValueError` if atemporal), `t_start/t_end/duration` (float seconds) and
`t_start_ticks/t_end_ticks/duration_ticks` (exact; raise `ValueError` if
atemporal), `dim_index(name)` (explicit or default `RangeIndex(size)`),
`dim_size(name)`.

Accessors (each a tiny helper object with `__getitem__`):

- `s.time[a:b]` — **seconds**: floats quantised once via `secs_to_ticks`; an
  `int` here means whole seconds (`seconds_to_ticks_exact`); `None` bounds =
  domain bounds; slice step forbidden (`ValueError`). Delegates to
  `slice_ticks`.
- `s.ticks[a:b]` — **ticks**: ints only (`TypeError` for floats); `None`
  bounds = domain bounds.
- `s.slice[dim, sel]` — positional selection along a named non-time dim;
  `sel: int | slice | list[int] | np.ndarray(int|bool)`. An `int` **drops**
  the dim (like numpy). Unknown dim on a Series → `DimensionError`. Using
  `"time"` here → `DimensionError` pointing at `.time`/`.ticks`.
- `s.sel[dim, label_or_list]` — label selection; requires `LabelIndex` on the
  dim (`DimensionError` otherwise). Scalar label drops the dim; list keeps it
  (absent labels are skipped per `LabelIndex.locate`).

Methods:

- `slice_ticks(a, b) -> Series` — time-slice: `tindex.slice(a, b)` → apply
  positional selection along the time axis, replace `indexes["time"]`.
- `shift(dt) -> Series` — `to_ticks` coercion (float = seconds, int = ticks);
  O(1); atemporal series raise `ValueError`.
- `concat(other) -> Series` (and `__add__`) — glue along time. Requires
  identical `dims`, matching data-presence, equal non-time indexes
  (`IncompatibleError` otherwise; `RangeIndex` defaults compared by size).
  Time logic delegates to `tindex.concat(other.tindex)` → apply
  `left_sel`/`right_sel` (None = take all) along the time axis, then
  `concat` the data.
- `take_dim(name, sel) -> Series` — positional selection along a named
  non-time dim; co-updates that dim's stored index via `.take(sel)` (dropped
  index for int sel); int sel removes the dim from `dims`.
- `interpolate(times, kind="linear", fill="clamp") -> np.ndarray` — ported
  semantics from `UniformSeries.interpolate`/`EventSeries.interpolate`:
  linear only; fill ∈ {"clamp","nan","error"}; query times float seconds or
  int ticks; works for `GridIndex` (grid = `sample_times()`) and `StampIndex`
  (grid = `abs_stamps`); `SpanIndex` → `TypeError`. Generalised to any time
  axis position via `np.moveaxis` (time axis of the result stays where it is
  in `dims`). Data converted with `to_numpy_f64`.
- `resample(new_sr, kind="linear") -> Series` — ported from
  `UniformSeries.resample` / `EventSeries.interpolate_uniform`: evaluate on a
  fresh phase-0 grid over the same declared domain, result gets a `GridIndex`.
- `equal(other) -> bool` (exact: dims, indexes `.equal`, data
  `array_equal`), `__eq__` delegating (NotImplemented for non-Series).
- `with_data(new_data) -> Series` — same shape check; `map_data(fn)` sugar.

Factory functions (also in `series.py`, re-exported at top level):

```python
uniform(data, sr, *, dims=None, t_start=0.0, phase=0.0)      # dims default (None,…,"time")
events(timestamps, values=None, *, dims=None, t_start=None, t_end=None)
spans(starts, ends, values=None, *, ids=None, t_start=None, t_end=None, dims=None)
wrap(data, dims=None, indexes=None)                          # atemporal; dims default all-None
```

`events`/`spans` with `values=None` build index-only Series (`data=None`,
`dims=("time",)`). Timestamp/bound arrays: float = seconds (quantised once),
int = ticks — matching the old `from_events`/`from_segments`.

## `Frame` — the pytree node (`tflib/frame.py`)

Frozen dataclass `Frame(entries, *, t_start=None, t_end=None)` with
`entries: Mapping[str, Series | Frame | Any]`.

Entry kinds:

- **temporal** — `Series` with a time dim, or a nested `Frame` that
  (recursively) contains temporal content; a nested frame holding only
  invariant entries (a pure metadata bundle) is itself invariant;
- **invariant** — atemporal `Series` (e.g. `mic_pos`), or any other Python
  scalar/object (`recording_id`, …). Raw numpy/torch tensors passed in are
  auto-wrapped: `wrap(x)` (all-anonymous dims).

Time anchoring (ported from `TimeFrame`): the frame owns the single absolute
anchor `t_start_ticks` + `dur_ticks` (hull of temporal children, inferred when
not given; validated to cover them; `(0, 0)` if none). Temporal children are
stored **relative** (constructor re-bases incoming absolute children by
`shift(-t_start_ticks)`); `frame[key]` hands back the child re-anchored to
absolute time (O(1)). Extracting then mutating never touches the parent —
everything is immutable. Internal `_from_local` classmethod bypasses
re-basing/validation.

**Tree-wide dim validation** (recursively over all Series leaves, `"time"`
excluded): for each dim name, all `RangeIndex` occurrences must agree in size;
mixing `RangeIndex` and `LabelIndex` occurrences of one dim is an error
(`DimensionError`); `LabelIndex` occurrences may differ in size and labels.

API:

- dict-like: `frame[key]`, `keys/values/items`, `in`, `len`, iteration.
- column ops: `select(keys)`, `drop(keys)`, `with_entry(name, value)`
  (hull expands as needed), `merge(other, overwrite=False)` (key collisions
  error unless overwrite; hull = union).
- time ops: `frame.time[a:b]`, `frame.ticks[a:b]`, `slice_ticks(a, b)`,
  `shift(dt)`, `concat(other)` / `+`.
  - slice: window must lie inside the frame domain (`DomainError`); each
    temporal child is clipped to its overlap with the window and dropped if
    disjoint; invariant entries pass through.
  - concat: `other` is glued so its domain starts at `self.t_end_ticks`;
    union of keys; temporal∧temporal → child concat (shift other by
    `self.dur_ticks - other.t_start_ticks` relative to frames' anchors, as in
    `TimeFrame.concat`); invariant∧invariant → must be `.equal`/`==`
    (`IncompatibleError` on conflict), keep one; one-sided entries carried
    over (temporal ones shifted); temporal∧invariant → `IncompatibleError`.
- dim ops (recursive over the tree, non-Series entries untouched, Series
  lacking the dim untouched):
  - `frame.slice[dim, sel]` — positional; `DimensionError` if no leaf has the
    dim, or if occurrences have unequal sizes (→ use `.sel`).
  - `frame.sel[dim, labels]` — all occurrences must carry `LabelIndex`.
  - `"time"` in either → `DimensionError`.
- `equal(other)` exact (anchors, key sets, children via `.equal`/`==`),
  `__eq__` delegating.
- `leaves() -> Iterator[tuple[str, Series]]` (dotted paths), `map_data(fn)`
  (apply to every Series leaf's data, e.g. numpy→torch).

Frame invariant (property-tested, composed across all leaf kinds):

    frame.ticks[a:b].concat(frame.ticks[b:c]) == frame.ticks[a:c]

## The motivating example

```python
tf = Frame({
    "audio":   uniform(audio_8xT, sr=44100, dims=("mic", "time")),
    "mic_pos": wrap(pos_8x3, dims=("mic", None)),
    "rps":     events(ts, rps_4xM, dims=("rotor", "time")),
    "rotor_pos": wrap(rpos_4x3, dims=("rotor", None)),
    "vad":     spans(starts, ends),
    "meta":    Frame({"recording_id": "FLY124"}),
})

sub = tf.slice["mic", 0]          # audio (T,), mic_pos (3,) — same mic 0
clip = tf.time[1.0:4.5]           # all temporal leaves cut, invariants kept
one = tf["rps"]                   # absolute-time Series; tf unchanged
```

## Code standards

- ruff + pyright (basic) must pass — a PostToolUse hook enforces this on
  every write. **No `# type: ignore`**: if pyright complains, change the
  design (make fields required, hoist optionality into constructors, narrow
  with `isinstance`) rather than suppressing.
- Frozen dataclasses, exact int64 tick arithmetic, no float tolerances
  outside the documented `phase`/`INDEX_EPSILON` cases.

## Non-goals (v1)

- Lazy / disk-backed storage; datetime64 interop; value-merging of
  coincident events; pytree registration with torch/jax (planned follow-up).
