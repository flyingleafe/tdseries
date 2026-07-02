# tdseries — agent guide

Read [`DESIGN.md`](./DESIGN.md) before touching library code: it is the
binding API contract (dimension/index model, Series and Frame semantics,
exactness invariants). [`README.md`](./README.md) is the human-facing
summary; [`ROADMAP.md`](./ROADMAP.md) records deliberately deferred
directions (CoordIndex, framed transforms over event series, lazy sources,
element-wise operator layer) — check it before proposing "new" features.

## Layout

| Path | Role |
|---|---|
| `src/tdseries/_ticks.py` | seconds↔int64-tick conversion boundary; `TICKS_PER_SECOND` |
| `src/tdseries/_array.py` | numpy/torch backend dispatch (`take`, `concat`, `array_equal`, ...) |
| `src/tdseries/errors.py` | `DomainError`, `IncompatibleError`, `DimensionError` |
| `src/tdseries/indexes.py` | the index hierarchy: `RangeIndex`, `LabelIndex` (ordinary dims); `GridIndex`, `StampIndex`, `SpanIndex` (time). All exact-tick cut/seam arithmetic lives here. |
| `src/tdseries/series.py` | `Series` leaf (data + dims + indexes) and factories `uniform`/`events`/`spans`/`wrap` |
| `src/tdseries/frame.py` | `Frame` pytree node: relative-anchor storage, hull, recursion of time/dim ops |
| `tests/` | hypothesis property tests; `strategies.py` draws exact int64 tick anchors (small and Unix-magnitude) |

## Load-bearing invariants (do not weaken)

1. `x.ticks[a:b].concat(x.ticks[b:c]) == x.ticks[a:c]` — exact slice-concat
   identity for every index kind and for frames; property-tested.
2. All time arithmetic is int64 ticks; the only floats are `GridIndex.phase`
   (bounded by one sample) and the non-integer-sr fallback (`INDEX_EPSILON`).
3. `shift` is O(1) everywhere: content is stored anchor-relative, only the
   scalar anchor moves. Frames store children relative to the frame anchor;
   extraction re-anchors a copy, never mutates.
4. No `# type: ignore` / `# noqa` — restructure until ruff+basedpyright pass
   (a PostToolUse hook may enforce this on every write).

## Workflow

- `uv sync` then `uv run pytest` (hypothesis suite, runs in tens of seconds).
- `nix develop` for the hooked dev shell (ruff, ruff-format, basedpyright).
- torch is an optional extra for runtime users (lazy import inside
  `_array.py` only, `src/tdseries` stays importable without it); it's a
  `dev` dependency-group member so basedpyright/tests can see it locally
  and in CI. `[tool.uv.sources]` pins it to the CPU-only wheel index so
  `uv sync` doesn't pull the CUDA/nvidia stack.
- CI (`.github/workflows/checks.yml`, reused by `ci.yml` and `publish.yml`)
  runs ruff + ruff-format (`nix flake check`), basedpyright, and pytest via
  `nix develop`; publishing to PyPI on tag push is gated on all three passing.
