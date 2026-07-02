# tflib — agent guide

Read [`DESIGN.md`](./DESIGN.md) before touching library code: it is the
binding API contract (dimension/index model, Series and Frame semantics,
exactness invariants). [`README.md`](./README.md) is the human-facing summary.

## Layout

| Path | Role |
|---|---|
| `src/tflib/_ticks.py` | seconds↔int64-tick conversion boundary; `TICKS_PER_SECOND` |
| `src/tflib/_array.py` | numpy/torch backend dispatch (`take`, `concat`, `array_equal`, ...) |
| `src/tflib/errors.py` | `DomainError`, `IncompatibleError`, `DimensionError` |
| `src/tflib/indexes.py` | the index hierarchy: `RangeIndex`, `LabelIndex` (ordinary dims); `GridIndex`, `StampIndex`, `SpanIndex` (time). All exact-tick cut/seam arithmetic lives here. |
| `src/tflib/series.py` | `Series` leaf (data + dims + indexes) and factories `uniform`/`events`/`spans`/`wrap` |
| `src/tflib/frame.py` | `Frame` pytree node: relative-anchor storage, hull, recursion of time/dim ops |
| `tests/` | hypothesis property tests; `strategies.py` draws exact int64 tick anchors (small and Unix-magnitude) |

## Load-bearing invariants (do not weaken)

1. `x.ticks[a:b].concat(x.ticks[b:c]) == x.ticks[a:c]` — exact slice-concat
   identity for every index kind and for frames; property-tested.
2. All time arithmetic is int64 ticks; the only floats are `GridIndex.phase`
   (bounded by one sample) and the non-integer-sr fallback (`INDEX_EPSILON`).
3. `shift` is O(1) everywhere: content is stored anchor-relative, only the
   scalar anchor moves. Frames store children relative to the frame anchor;
   extraction re-anchors a copy, never mutates.
4. No `# type: ignore` / `# noqa` — restructure until ruff+pyright pass
   (a PostToolUse hook may enforce this on every write).

## Workflow

- `uv sync` then `uv run pytest` (hypothesis suite, runs in tens of seconds).
- `nix develop` for the hooked dev shell (ruff, ruff-format, pyright).
- torch is an optional extra; keep `src/tflib` importable without it
  (lazy import inside `_array.py` only).
