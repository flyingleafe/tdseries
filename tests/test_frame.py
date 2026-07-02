"""Frame-level time algebra: slice/concat/shift composed over grid + stamp +
span leaves, invariant entries, extraction absolutization, column ops, and
nested-frame recursion."""

from __future__ import annotations

import operator

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdseries import (
    Frame,
    IncompatibleError,
    Series,
    events,
    spans,
    uniform,
    wrap,
)
from tdseries.errors import DomainError

from .strategies import (
    cut_points_ticks,
    sample_rates,
    sorted_stamps_with_values,
    span_bounds,
    tick_anchors,
)

# ---------------------------------------------------------------------------
# Strategy: a frame whose temporal children share one arbitrary [t0, t1),
# plus an atemporal Series and a scalar entry.
# ---------------------------------------------------------------------------


@st.composite
def mixed_frame(draw) -> Frame:
    sr = draw(sample_rates)
    n = draw(st.integers(min_value=4, max_value=40))
    t0 = draw(tick_anchors)
    samples = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=np.float64,
    )
    audio = uniform(samples, sr=sr, t_start=t0)
    dur = audio.duration_ticks
    t1 = audio.t_end_ticks

    m = draw(st.integers(min_value=0, max_value=8))
    stamps, values = draw(sorted_stamps_with_values(t0, dur, m))
    rps = events(stamps, values, t_start=t0, t_end=t1)

    k = draw(st.integers(min_value=0, max_value=3))
    starts, ends = draw(span_bounds(t0, dur, k))
    vad = spans(starts, ends, t_start=t0, t_end=t1)

    pos = wrap(
        np.asarray(
            draw(
                st.lists(
                    st.floats(
                        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
                    ),
                    min_size=3,
                    max_size=3,
                )
            ),
            dtype=np.float64,
        )
    )
    rid = draw(st.sampled_from(["FLY124", "rec-2", 7]))
    return Frame({"audio": audio, "rps": rps, "vad": vad, "pos": pos, "recording_id": rid})


@st.composite
def _cuts(draw, tf: Frame, k: int) -> list[int]:
    return draw(cut_points_ticks(tf.t_start_ticks, tf.t_end_ticks, k))


# ---------------------------------------------------------------------------
# Slice-concat identity composed across all leaf kinds + invariants
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(mixed_frame())
def test_full_domain_slice_is_equal(tf: Frame):
    assert tf.ticks[tf.t_start_ticks : tf.t_end_ticks].equal(tf)
    assert tf.ticks[:].equal(tf)


@settings(deadline=None, max_examples=100)
@given(mixed_frame(), st.data())
def test_slice_concat_identity(tf: Frame, data):
    a, b, c = data.draw(_cuts(tf, 3))
    assert tf.ticks[a:b].concat(tf.ticks[b:c]).equal(tf.ticks[a:c])


@settings(deadline=None, max_examples=60)
@given(mixed_frame(), st.data())
def test_many_cuts_rejoin(tf: Frame, data):
    k = data.draw(st.integers(min_value=2, max_value=5))
    pts = [tf.t_start_ticks, *data.draw(_cuts(tf, k)), tf.t_end_ticks]
    parts = [tf.ticks[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(tf)


@settings(deadline=None, max_examples=100)
@given(mixed_frame(), st.data())
def test_invariants_preserved_unchanged_by_slice(tf: Frame, data):
    a, b = data.draw(_cuts(tf, 2))
    sliced = tf.ticks[a:b]
    assert isinstance(sliced["pos"], Series)
    assert sliced["pos"].equal(tf["pos"])
    assert sliced["recording_id"] == tf["recording_id"]


@settings(deadline=None, max_examples=100)
@given(mixed_frame(), st.data())
def test_shift_roundtrip(tf: Frame, data):
    dt = data.draw(st.integers(min_value=-(10**12), max_value=10**12))
    assert tf.shift(dt).shift(-dt).equal(tf)


# ---------------------------------------------------------------------------
# Extraction absolutization + immutability
# ---------------------------------------------------------------------------


def _toy_frame(t0: int = 0) -> Frame:
    audio = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=t0)
    rps = events(
        np.array([t0 + 100_000_000, t0 + 500_000_000], dtype=np.int64),
        np.array([1.0, 2.0]),
        t_start=t0,
        t_end=t0 + 1_000_000_000,
    )
    return Frame({"audio": audio, "rps": rps, "recording_id": "FLY124"})


def test_extracted_child_carries_absolute_t_start():
    tf = _toy_frame(t0=10_000_000_000)
    shifted = tf.shift(5_000_000_000)
    child = shifted["audio"]
    assert child.t_start_ticks == 15_000_000_000
    assert child.t_end_ticks == 16_000_000_000


def test_extraction_does_not_mutate_frame():
    tf = _toy_frame(t0=10_000_000_000)
    snapshot = _toy_frame(t0=10_000_000_000)
    child = tf["audio"]
    # Mutating (rebuilding) the extracted child leaves the parent untouched.
    moved = child.shift(7_000_000_000)
    assert moved.t_start_ticks == 17_000_000_000
    assert tf["audio"].t_start_ticks == 10_000_000_000
    assert tf.equal(snapshot)


def test_shift_does_not_mutate_original():
    tf = _toy_frame(t0=0)
    snapshot = _toy_frame(t0=0)
    _ = tf.shift(3_000_000_000)
    assert tf.equal(snapshot)


def test_extracted_child_equals_standalone_series():
    t0 = 2_000_000_000
    audio = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=t0)
    tf = Frame({"audio": audio})
    assert tf["audio"].equal(audio)


# ---------------------------------------------------------------------------
# frame.time[...] vs frame.ticks[...]
# ---------------------------------------------------------------------------


def test_time_seconds_equals_ticks():
    audio = uniform(np.zeros(60), sr=10, t_start=0)  # [0, 6) s
    tf = Frame({"audio": audio, "recording_id": "x"})
    assert tf.time[1.0:4.5].equal(tf.ticks[1_000_000_000:4_500_000_000])


def test_time_int_bounds_are_whole_seconds():
    audio = uniform(np.zeros(60), sr=10, t_start=0)
    tf = Frame({"audio": audio})
    assert tf.time[1:4].equal(tf.ticks[1_000_000_000:4_000_000_000])


def test_ticks_accessor_rejects_float_bounds():
    tf = _toy_frame()
    with pytest.raises(TypeError):
        tf.ticks[0.0:0.5]  # type: ignore[misc]


def test_slice_outside_domain_raises():
    tf = _toy_frame(t0=0)
    with pytest.raises(DomainError):
        tf.ticks[-500_000_000:500_000_000]


# ---------------------------------------------------------------------------
# Concat: invariant conflicts, one-sided entries, dunder
# ---------------------------------------------------------------------------


def _frame_with_scalar(t0: int, rid: str) -> Frame:
    audio = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=t0)
    return Frame({"audio": audio, "recording_id": rid})


def test_concat_conflicting_scalar_entries_raises():
    a = _frame_with_scalar(0, "A")
    b = _frame_with_scalar(1_000_000_000, "B")
    with pytest.raises(IncompatibleError):
        a.concat(b)


def test_concat_equal_scalar_entries_kept():
    a = _frame_with_scalar(0, "A")
    b = _frame_with_scalar(1_000_000_000, "A")
    joined = a.concat(b)
    assert joined["recording_id"] == "A"
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 2_000_000_000


def test_concat_temporal_vs_invariant_raises():
    a = Frame({"x": uniform(np.arange(10.0), sr=10, t_start=0)})
    b = Frame(
        {
            "audio": uniform(np.arange(10.0), sr=10, t_start=0),
            "x": 42,
        }
    )
    with pytest.raises(IncompatibleError):
        a.concat(b)


def test_concat_one_sided_entries_carried_over():
    a = Frame({"audio": uniform(np.arange(10.0), sr=10, t_start=0), "id": "s"})
    b = Frame(
        {
            "rps": events(
                np.array([100], dtype=np.int64), np.array([1.0]), t_start=0, t_end=1_000_000_000
            )
        }
    )
    joined = a.concat(b)
    assert set(joined.keys()) == {"audio", "rps", "id"}
    # The one-sided temporal entry is shifted to start at the seam.
    assert joined["rps"].t_start_ticks == 1_000_000_000


def test_plus_operator_is_not_concat():
    # ``+`` is reserved for future aligned element-wise ops; concatenation is
    # the named ``.concat`` method only.
    a = _frame_with_scalar(0, "A")
    b = _frame_with_scalar(1_000_000_000, "A")
    with pytest.raises(TypeError):
        operator.add(a, b)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_disjoint_keys_union():
    a = Frame({"audio": uniform(np.arange(10.0), sr=10, t_start=0)})
    b = Frame(
        {"rps": events(np.array([100], dtype=np.int64), np.array([1.0]), t_start=0, t_end=500)}
    )
    merged = a.merge(b)
    assert set(merged.keys()) == {"audio", "rps"}


def test_merge_collision_raises():
    a = _frame_with_scalar(0, "A")
    b = _frame_with_scalar(0, "B")
    with pytest.raises(IncompatibleError):
        a.merge(b)


def test_merge_collision_overwrite_takes_other():
    a = _frame_with_scalar(0, "A")
    b = _frame_with_scalar(0, "B")
    merged = a.merge(b, overwrite=True)
    assert merged["recording_id"] == "B"


def test_merge_hull_is_union():
    a = Frame({"x": uniform(np.arange(10.0), sr=10, t_start=0)})
    b = Frame({"y": uniform(np.arange(10.0), sr=10, t_start=2_000_000_000)})
    merged = a.merge(b)
    assert merged.t_start_ticks == 0
    assert merged.t_end_ticks == 3_000_000_000


# ---------------------------------------------------------------------------
# select / drop / with_entry
# ---------------------------------------------------------------------------


def test_select_keeps_subset():
    tf = _toy_frame()
    sub = tf.select(["audio"])
    assert set(sub.keys()) == {"audio"}
    assert sub["audio"].equal(tf["audio"])


def test_select_missing_raises():
    tf = _toy_frame()
    with pytest.raises(KeyError):
        tf.select(["audio", "missing"])


def test_drop_removes_keys():
    tf = _toy_frame()
    sub = tf.drop(["rps"])
    assert set(sub.keys()) == {"audio", "recording_id"}


def test_with_entry_adds_and_expands_hull():
    tf = _toy_frame(t0=0)  # domain [0, 1 s)
    late = uniform(np.zeros(5), sr=10, t_start=5_000_000_000)
    extended = tf.with_entry("late", late)
    assert "late" in extended
    assert extended.t_start_ticks == 0
    assert extended.t_end_ticks == 5_500_000_000


def test_with_entry_scalar_keeps_hull():
    tf = _toy_frame(t0=0)
    extended = tf.with_entry("snr", -10.0)
    assert extended["snr"] == -10.0
    assert extended.t_start_ticks == tf.t_start_ticks
    assert extended.t_end_ticks == tf.t_end_ticks


def test_with_entry_auto_wraps_raw_arrays():
    tf = _toy_frame(t0=0)
    extended = tf.with_entry("pos", np.array([1.0, 2.0, 3.0]))
    entry = extended["pos"]
    assert isinstance(entry, Series)
    assert entry.dims == (None,)


# ---------------------------------------------------------------------------
# Dict-like protocol
# ---------------------------------------------------------------------------


def test_dict_protocol():
    tf = _toy_frame()
    assert len(tf) == 3
    assert set(tf) == {"audio", "rps", "recording_id"}
    assert "audio" in tf
    assert "missing" not in tf
    assert {k for k, _ in tf.items()} == set(tf.keys())
    assert len(list(tf.values())) == 3


# ---------------------------------------------------------------------------
# Nested Frame recursion
# ---------------------------------------------------------------------------


def _nested_frame(t0: int = 0) -> Frame:
    inner = Frame(
        {
            "audio": uniform(np.arange(10, dtype=np.float64), sr=10, t_start=t0),
            "recording_id": "inner-rec",
        }
    )
    outer_audio = uniform(np.arange(20, dtype=np.float64), sr=20, t_start=t0)
    return Frame({"inner": inner, "audio": outer_audio})


def test_nested_frame_slices_with_parent():
    tf = _nested_frame(t0=0)
    sliced = tf.ticks[200_000_000:700_000_000]
    inner = sliced["inner"]
    assert isinstance(inner, Frame)
    assert inner.t_start_ticks == 200_000_000
    assert inner.t_end_ticks == 700_000_000
    expected = uniform(np.arange(10, dtype=np.float64), sr=10, t_start=0).ticks[
        200_000_000:700_000_000
    ]
    assert inner["audio"].equal(expected)


def test_nested_frame_shifts_with_parent():
    tf = _nested_frame(t0=0)
    shifted = tf.shift(1_000_000_000)
    inner = shifted["inner"]
    assert isinstance(inner, Frame)
    assert inner.t_start_ticks == 1_000_000_000
    assert inner["audio"].t_start_ticks == 1_000_000_000
    # Round trip restores the original exactly.
    assert shifted.shift(-1_000_000_000).equal(tf)


def test_nested_frame_slice_concat_identity():
    tf = _nested_frame(t0=0)
    a, b, c = 100_000_000, 430_000_000, 900_000_000
    assert tf.ticks[a:b].concat(tf.ticks[b:c]).equal(tf.ticks[a:c])


def test_nested_frame_invariants_survive_slicing():
    tf = _nested_frame(t0=0)
    sliced = tf.ticks[200_000_000:700_000_000]
    inner = sliced["inner"]
    assert isinstance(inner, Frame)
    assert inner["recording_id"] == "inner-rec"


def test_leaves_yields_dotted_paths():
    tf = _nested_frame(t0=0)
    paths = {path for path, _ in tf.leaves()}
    assert paths == {"audio", "inner.audio"}


# ---------------------------------------------------------------------------
# Equality dunder
# ---------------------------------------------------------------------------


def test_eq_delegates_to_equal():
    a = _toy_frame()
    b = _toy_frame()
    assert (a == b) == a.equal(b)
    assert a == b


def test_eq_non_frame_is_notimplemented():
    tf = _toy_frame()
    assert tf.__eq__(42) is NotImplemented


def test_equal_differs_on_scalar_entry():
    a = _frame_with_scalar(0, "A")
    b = _frame_with_scalar(0, "B")
    assert not a.equal(b)


# ---------------------------------------------------------------------------
# Invariant-only nested frames (pure metadata bundles)
# ---------------------------------------------------------------------------


def test_metadata_frame_is_invariant_under_time_ops():
    """A nested frame with no temporal content survives any time window
    unchanged (it must not be treated as a [0, 0) temporal child)."""
    audio = uniform(np.arange(16000, dtype=np.float64), sr=16000, t_start=5.0)
    meta = Frame({"recording_id": "FLY124", "split": "valid"})
    tf = Frame({"audio": audio, "meta": meta})
    assert tf.t_start_ticks == audio.t_start_ticks  # meta does not pull hull to 0

    clip = tf.time[5.2:5.7]
    assert "meta" in clip
    inner = clip["meta"]
    assert isinstance(inner, Frame)
    assert inner["recording_id"] == "FLY124"

    joined = tf.time[5.0:5.5].concat(tf.time[5.5:6.0])
    assert joined["meta"].equal(meta)
