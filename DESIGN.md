# tdseries — design contract

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

Index families (implemented in `tdseries/indexes.py`, already written):

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

## `Series` — the leaf (`tdseries/series.py`)

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
index. Use `tdseries._array.is_tensor/take/concat/array_equal/to_numpy_f64` for
all data manipulation (they dispatch over any Array-API backend — numpy, torch,
JAX, … — via `array-api-compat`; the index layer stays numpy-`int64`).

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
- `concat(other) -> Series` — glue along time. Requires
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

## `Frame` — the pytree node (`tdseries/frame.py`)

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
  `shift(dt)`, `concat(other)`.
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

## Operators (decided 2026-07-02)

Arithmetic dunders (`+`, `*`, ...) are **reserved for aligned element-wise
operations** (mixing, gain), matching numpy/xarray intuition — a video-editor
mix should read `0.5 * a + 0.3 * b`.  Concatenation is the named `.concat()`
method only; `__add__` is NOT concat.  Until the align/apply layer lands
(below), the arithmetic dunders are simply absent.

## Exact rational sample rates (decided 2026-07-02)

`GridIndex` stores its rate as a normalized integer fraction
`(sr_num, sr_den)` — samples per second — not a float.  Motivation: framed
transforms (STFT with hop `h`) produce rates `sr / h` that are non-integer
but exactly rational; a float rate would demote them to an epsilon-tolerance
path.  With rational rates every grid computation is exact integer
arithmetic (`divmod` against `TICKS_PER_SECOND * sr_den`), the epsilon
fallback is deleted outright, and **isomorphism roundtrips preserve the index
exactly**:

    inverse_spec(forward_spec(idx)).equal(idx)      # wav -> STFT -> iSTFT -> wav

Constructors accept `int`, integral `float`, `fractions.Fraction`, or a
`(num, den)` tuple; non-integral floats are rejected (pass the exact
fraction instead).

## Transforms, alignment, streaming — v1.1 design (agreed, not yet implemented)

Regularizer domains used to pressure-test these decisions: the drone project
(audio+telemetry), video-editor backend, finance (ticks/bars/as-of joins),
physiological monitoring (multi-rate biosignals + categorical stages),
football analytics (LabelIndex joins, ragged player dims), and server
observability (logs/traces as stamps/spans, windowed aggregation).

### Unary framed transforms

The library owns no DSP — only the *time bookkeeping* of transforms.  All
practically relevant grid-warping transforms (STFT, conv/pool stacks,
resamplers) share one structure: output step `k` depends on the input span
`[k*hop - left, k*hop - left + win)`.  This is captured by a declarative
spec:

```python
spec = Framed(win=1024, hop=256, center=True)    # causal: left=win-1, right=0
latents = audio.transform(stft_fn, time=spec, dims=("mic", "freq", "time"))
```

`transform` runs the user's arbitrary `data -> data` function; the spec is
used for exactly two derived quantities:

1. **forward map** — `GridIndex(rate, anchor, phase) ->
   GridIndex(rate/hop, anchor', phase')`; frame-center alignment is
   expressed through the existing `phase` mechanism (center=True gives
   output phase -0.5).
2. **pullback** — `spec.required_input(a, b)`: the input tick window needed
   to compute output window `[a, b)` (receptive-field arithmetic).  This
   powers patch inference and streaming.

Specs compose (`spec_a >> spec_b`, standard stride/receptive-field
composition), so a deep encoder's time semantics is derived, never
hand-computed.  Invertible pairs (STFT/iSTFT) must satisfy the roundtrip
identity above exactly.  New non-time dims (`"freq"`) arrive as ordinary
named dims (bare `RangeIndex` in v1; the numeric `CoordIndex` is deferred —
see `ROADMAP.md`).

**v1 scope (decided 2026-07-02): `transform` accepts `GridIndex` inputs
only** (uniform -> uniform).  Sliding-window operators over non-uniform
event series (moving averages over ticks, event-rate counters) go through
`resample_on(grid)` first; that detour is semantically complete but
inefficient for sparse events, so a native `FramedEvents` generalization is
on the roadmap (`ROADMAP.md` § Framed transforms over non-uniform series).

### N-ary operations = align, then broadcast by name, then ufunc

Three orthogonal primitives instead of an n-ary op zoo:

1. `align(*series, domain="intersection"|"union", to=<series|index>,
   fill=...)` — the ONLY place where domains/grids are reconciled.
   Union+fill=0 is the video-editor clip mix; `resample_on(other,
   kind="previous")` (zero-order hold) is simultaneously the DAW automation
   curve and the finance as-of join.  `interpolate` gains
   `kind="previous"|"nearest"` alongside `"linear"`.
2. **Named-dim broadcasting**: dims are matched by NAME, not position —
   `(mic, time) * (time,)` broadcasts, `(mic, time) * (rotor, time)`
   outer-broadcasts to `(mic, rotor, time)`.
3. **Strict element-wise application**: `tdseries.apply(fn, *series)` and the
   arithmetic dunders require *identical* time indexes and raise
   `IncompatibleError` otherwise.  No silent xarray-style intersection —
   the finance regularizer forbids implicit data loss; magic never crosses
   domain boundaries, only explicit `align` does.

### Streaming

Streaming introduces no new ontology; the exactness algebra is the
correctness proof for chunking:

    f(x).ticks[a:b] == f(x.ticks[a-left : b+right]).ticks[a:b]

with `left`/`right` from the spec pullback — property-testable.  Three thin
pieces make it real:

1. **Lazy leaf protocol** (v1 scope: protocol only): `Series.data` requires
   only `shape`, `dtype`, `ndim`, and `__getitem__` returning an ndarray.
   numpy views already make slicing zero-copy and `np.memmap` works today;
   video decoders (PyAV/decord) plug in later behind a small adapter.
2. **Stateless pull execution**: `frame.stream(start, chunk)` yields
   successive `frame.ticks[t : t+chunk)`; a transform in the pipeline pulls
   its `required_input` window per chunk.  Stateful streaming (RNN carry)
   is explicitly out of scope — that state belongs to the model.
3. **Streaming mixing** is per-chunk `align(union, fill=0)` — no special
   case.

## The motivating example

```python
frame = Frame({
    "audio":   uniform(audio_8xT, sr=44100, dims=("mic", "time")),
    "mic_pos": wrap(pos_8x3, dims=("mic", None)),
    "rps":     events(ts, rps_4xM, dims=("rotor", "time")),
    "rotor_pos": wrap(rpos_4x3, dims=("rotor", None)),
    "vad":     spans(starts, ends),
    "meta":    Frame({"recording_id": "FLY124"}),
})

sub = frame.slice["mic", 0]          # audio (T,), mic_pos (3,) — same mic 0
clip = frame.time[1.0:4.5]           # all temporal leaves cut, invariants kept
one = frame["rps"]                   # absolute-time Series; frame unchanged
```

## Code standards

- ruff + pyright (basic) must pass — a PostToolUse hook enforces this on
  every write. **No `# type: ignore`**: if pyright complains, change the
  design (make fields required, hoist optionality into constructors, narrow
  with `isinstance`) rather than suppressing.
- Frozen dataclasses, exact int64 tick arithmetic, no float tolerances
  outside the documented `phase`/`INDEX_EPSILON` cases.

## Non-goals (v1)

- Lazy / disk-backed storage beyond the array protocol; datetime64 interop;
  value-merging of coincident events; pytree registration with torch/jax.
  Deferred directions live in [`ROADMAP.md`](./ROADMAP.md).
