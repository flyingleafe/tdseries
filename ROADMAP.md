# tdseries — future plans

Deliberately deferred design directions. Each entry states the motivation,
the shape the design already leaves for it, and why it is not in v1.
Decisions here were pressure-tested against the regularizer domains listed in
`DESIGN.md` (drone audio, video editing, finance, physiological monitoring,
football analytics, server observability).

## CoordIndex — numeric coordinates for non-time dims

**What.** A fourth index kind mapping positions to *ordered real
coordinates* (STFT bin-center Hz, image rows in mm, beamforming steering
angles), completing the taxonomy: `RangeIndex` (positions), `LabelIndex`
(discrete labels), `TimeIndex` (ticks), `CoordIndex` (ordered reals).

```python
stft = audio.transform(stft_fn, time=spec,
                       dims=("mic", "freq", "time"),
                       freq=CoordIndex(bin_hz))
band = stft.coord["freq", 300.0:3400.0]      # select by physical Hz
```

Like time and labels, it would let leaves share a dim at different
resolutions (linear-frequency and mel spectrograms both answering one
Hz-range query).

**Why deferred.** Float range queries reintroduce the tolerance/boundary
conventions that ticks were invented to eliminate; acceptable for frequency
(nobody round-trips Hz) but it is design surface with no consumer yet. The
per-leaf index-translation machinery, the accessor pattern, and the
ragged-sizes rule all transfer directly when it lands.

## Framed transforms over non-uniform series

**What.** v1 `Framed`/`transform` supports `GridIndex` inputs only
(uniform -> uniform). Sliding-window operators over *event* series —
moving averages over trade ticks, event-rate counters over log streams,
windowed HRV over heartbeats — must currently detour through
`resample_on(grid)` first, which is wasteful when events are sparse or the
window/step structure does not align with any natural grid.

**Design sketch.** A `FramedEvents(win, hop_or_trigger)` spec whose forward
map produces a `GridIndex` (fixed hop) or `StampIndex` (per-event windows,
as in "average of the last 100 ms at every event") and whose pullback is a
tick-window query answered by `searchsorted` — the receptive-field algebra
is identical, only the position<->tick map changes. The chunking-correctness
identity (`DESIGN.md` § Streaming) carries over unchanged.

**Why deferred.** One clean case (grid) first; the composition
`resample_on >> grid transform` covers the semantics, just not the
efficiency.

## Lazy leaf sources beyond memmap

v1 defines the minimal array protocol (`shape`, `dtype`, `ndim`,
`__getitem__` -> ndarray); `np.ndarray` views and `np.memmap` already
qualify. Planned adapters: video decoders (PyAV / decord) for the
video-editor case, chunked columnar stores (parquet/zarr) for finance and
observability.

## Element-wise operator layer

Arithmetic dunders are reserved (see `DESIGN.md` § Operators) and absent
until the `align` / named-dim-broadcast / `apply` layer is implemented.
Implementation order: `interpolate(kind="previous"|"nearest")` ->
`resample_on` -> `align` -> `apply` + dunders.

## Stateful streaming

Stateless pull execution is the v1 model. Carrying state across chunks
(RNN hidden state, streaming-conv context caches) stays out of scope: that
state belongs to the model, not the time algebra. Revisit only if a
concrete consumer shows the boundary is wrong.

## Pytree registration (torch / jax)

Register `Frame`/`Series` with `torch.utils._pytree` (and optionally jax)
so frames pass through `default_collate` and `tree_map` natively. Needs
care: anchors and indexes are context, data is leaves.

## datetime64 / wall-clock interop

Thin converters between anchors and `numpy.datetime64[ns]` / stdlib
`datetime`. Ticks stay the substrate; calendars and timezones remain a
*view* concern (the finance regularizer wants exchange calendars — that
belongs in user code or a separate helper, never in the core algebra).
