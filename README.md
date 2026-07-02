# tdseries

Immutable pytree frames of tensors with **named, indexed dimensions**, where
**time** is a first-class dimension stored in exact int64 ticks (nanoseconds).

Built for bundling media signals with co-recorded telemetry *and* the static
geometry that shares their non-time dimensions — e.g. multichannel audio
`(mic, time)` together with microphone positions `(mic, 3)`, rotor speeds
`(rotor, time)` together with rotor positions `(rotor, 3)`:

```python
import tdseries as td

frame = td.Frame({
    "audio":     td.uniform(audio_8xT, sr=44100, dims=("mic", "time")),
    "mic_pos":   td.wrap(pos_8x3, dims=("mic", None)),
    "rps":       td.events(stamps, rps_4xM, dims=("rotor", "time")),
    "rotor_pos": td.wrap(rpos_4x3, dims=("rotor", None)),
    "vad":       td.spans(starts, ends),
    "recording_id": "FLY124",
})

sub  = frame.slice["mic", 0]     # audio -> (T,), mic_pos -> (3,): same mic
clip = frame.time[1.0:4.5]       # seconds; all temporal leaves cut exactly
raw  = frame.ticks[a:b]          # exact int64 tick window
rps  = frame["rps"]              # self-contained absolute-time Series
```

## The model

A *dimension* is a name plus a domain; every tensor axis carrying the name has
an **index** mapping positions into that domain. Slices are expressed in
domain coordinates and translated per-tensor — which is how tensors of
*different lengths* share one dimension:

| Index | Domain | Example |
|---|---|---|
| `RangeIndex` (default) | positions | `mic`, `rotor` |
| `LabelIndex` | hashable labels | mic serial numbers |
| `GridIndex` | int64 ticks, uniform grid + sub-sample phase | audio |
| `StampIndex` | int64 ticks, sorted point events | RPS, IMU |
| `SpanIndex` | int64 ticks, half-open intervals with identity tags | VAD |

Time is exact: all anchors and cut points are int64 tick counts, `shift` is
O(1), and the algebra satisfies, at the sample level and without tolerances,

```
x.ticks[a:b].concat(x.ticks[b:c]) == x.ticks[a:c]
```

for any tick cut points — including sub-sample cuts through audio and cuts
through the middle of a VAD span (split spans re-merge via identity tags).

Everything is immutable. Extracting an entry from a frame hands back a
self-contained series re-anchored to absolute time; the frame is untouched.

See [`DESIGN.md`](./DESIGN.md) for the full contract.

## Development

```sh
nix develop        # or: uv sync
uv run pytest
```

`nix develop` provides Python + uv and pre-commit hooks (ruff, ruff-format,
pyright). Data leaves may be numpy arrays or torch tensors (torch optional:
`uv sync --extra torch`).

## Lineage

Extracted from the `utils.data` module of the harmonic-noise-suppression
research project; the tick/phase arithmetic and the slice/concat/shift
invariants are ported from there, the named-dimension frame model is new.
