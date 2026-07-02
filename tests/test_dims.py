"""Named-dimension mechanics on ``Series`` and ``Frame``: positional
selection (``.slice``), label selection (``.sel``), the reserved ``"time"``
dim, and tree-wide dim validation at ``Frame`` construction."""

from __future__ import annotations

import numpy as np
import pytest

from tflib import (
    DimensionError,
    Frame,
    LabelIndex,
    Series,
    events,
    spans,
    uniform,
    wrap,
)

# ---------------------------------------------------------------------------
# The DESIGN.md motivating example, verbatim
# ---------------------------------------------------------------------------


def _motivating_frame() -> Frame:
    n_time = 44100 * 5  # 5 s of audio
    audio_8xT = np.zeros((8, n_time), dtype=np.float32)
    pos_8x3 = np.arange(24, dtype=np.float64).reshape(8, 3)
    ts = np.arange(10, dtype=np.int64) * 500_000_000  # 0 .. 4.5 s
    rps_4xM = np.arange(40, dtype=np.float64).reshape(4, 10)
    rpos_4x3 = np.arange(12, dtype=np.float64).reshape(4, 3)
    starts = np.array([500_000_000, 2_000_000_000], dtype=np.int64)
    ends = np.array([1_500_000_000, 3_000_000_000], dtype=np.int64)
    return Frame(
        {
            "audio": uniform(audio_8xT, sr=44100, dims=("mic", "time")),
            "mic_pos": wrap(pos_8x3, dims=("mic", None)),
            "rps": events(ts, rps_4xM, dims=("rotor", "time"), t_start=0, t_end=5_000_000_000),
            "rotor_pos": wrap(rpos_4x3, dims=("rotor", None)),
            "vad": spans(starts, ends, t_start=0, t_end=5_000_000_000),
            "meta": Frame({"recording_id": "FLY124"}),
        }
    )


def test_motivating_example_mic_int_selector():
    tf = _motivating_frame()
    n_time = tf["audio"].dim_size("time")
    sub = tf.slice["mic", 0]
    assert sub["audio"].shape == (n_time,)
    assert sub["audio"].dims == ("time",)
    assert sub["mic_pos"].shape == (3,)
    assert sub["mic_pos"].dims == (None,)
    # Same mic 0: positions row 0 survived.
    np.testing.assert_array_equal(np.asarray(sub["mic_pos"].data), [0.0, 1.0, 2.0])
    # rotor-dim and time-only leaves are untouched.
    assert sub["rps"].shape == (4, 10)
    assert sub["rotor_pos"].shape == (4, 3)
    assert sub["vad"].equal(tf["vad"])


def test_motivating_example_mic_slice_selector():
    tf = _motivating_frame()
    n_time = tf["audio"].dim_size("time")
    sub = tf.slice["mic", 1:3]
    assert sub["audio"].shape == (2, n_time)
    assert sub["audio"].dims == ("mic", "time")
    assert sub["mic_pos"].shape == (2, 3)
    assert sub["mic_pos"].dims == ("mic", None)
    np.testing.assert_array_equal(
        np.asarray(sub["mic_pos"].data), [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    )


def test_motivating_example_rotor_selector():
    tf = _motivating_frame()
    sub = tf.slice["rotor", 2]
    assert sub["rps"].shape == (10,)
    assert sub["rps"].dims == ("time",)
    assert sub["rotor_pos"].shape == (3,)
    # mic leaves untouched.
    assert sub["audio"].shape == tf["audio"].shape


# ---------------------------------------------------------------------------
# Series-level: integer selector drops the dim, slice keeps it
# ---------------------------------------------------------------------------


def test_series_int_selector_drops_dim():
    s = uniform(np.arange(40, dtype=np.float64).reshape(4, 10), sr=10, dims=("mic", "time"))
    sub = s.slice["mic", 0]
    assert sub.dims == ("time",)
    assert sub.shape == (10,)
    np.testing.assert_array_equal(np.asarray(sub.data), np.arange(10.0))


def test_series_slice_selector_keeps_dim():
    s = uniform(np.arange(40, dtype=np.float64).reshape(4, 10), sr=10, dims=("mic", "time"))
    sub = s.slice["mic", 1:3]
    assert sub.dims == ("mic", "time")
    assert sub.shape == (2, 10)


def test_series_list_selector_keeps_dim():
    s = uniform(np.arange(40, dtype=np.float64).reshape(4, 10), sr=10, dims=("mic", "time"))
    sub = s.slice["mic", [0, 3]]
    assert sub.dims == ("mic", "time")
    assert sub.shape == (2, 10)


def test_series_slice_time_raises():
    s = uniform(np.arange(10.0), sr=10)
    with pytest.raises(DimensionError):
        s.slice["time", 0]


def test_series_slice_unknown_dim_raises():
    s = uniform(np.arange(10.0), sr=10)
    with pytest.raises(DimensionError):
        s.slice["mic", 0]


# ---------------------------------------------------------------------------
# Series-level: .sel label selection
# ---------------------------------------------------------------------------


def test_series_sel_scalar_label_drops_dim():
    w = wrap(
        np.array([1.0, 2.0, 3.0]),
        dims=("mic",),
        indexes={"mic": LabelIndex(("a", "b", "c"))},
    )
    sub = w.sel["mic", "b"]
    assert sub.dims == ()
    assert float(sub.data) == 2.0


def test_series_sel_list_keeps_dim():
    w = wrap(
        np.array([1.0, 2.0, 3.0]),
        dims=("mic",),
        indexes={"mic": LabelIndex(("a", "b", "c"))},
    )
    sub = w.sel["mic", ["a", "c"]]
    assert sub.dims == ("mic",)
    assert sub.shape == (2,)
    idx = sub.dim_index("mic")
    assert isinstance(idx, LabelIndex)
    assert idx.labels == ("a", "c")
    np.testing.assert_array_equal(np.asarray(sub.data), [1.0, 3.0])


def test_series_sel_absent_labels_skipped():
    w = wrap(
        np.array([1.0, 2.0, 3.0]),
        dims=("mic",),
        indexes={"mic": LabelIndex(("a", "b", "c"))},
    )
    sub = w.sel["mic", ["c", "z"]]
    assert sub.shape == (1,)
    np.testing.assert_array_equal(np.asarray(sub.data), [3.0])


def test_series_sel_without_label_index_raises():
    w = wrap(np.array([1.0, 2.0, 3.0]), dims=("mic",))
    with pytest.raises(DimensionError):
        w.sel["mic", "b"]


def test_series_sel_time_raises():
    s = uniform(np.arange(10.0), sr=10)
    with pytest.raises(DimensionError):
        s.sel["time", "x"]


# ---------------------------------------------------------------------------
# Frame-level: reserved "time" and unknown dims
# ---------------------------------------------------------------------------


def test_frame_slice_time_raises():
    tf = Frame({"audio": uniform(np.arange(10.0), sr=10)})
    with pytest.raises(DimensionError):
        tf.slice["time", 0]


def test_frame_sel_time_raises():
    tf = Frame({"audio": uniform(np.arange(10.0), sr=10)})
    with pytest.raises(DimensionError):
        tf.sel["time", "x"]


def test_frame_slice_unknown_dim_raises():
    tf = Frame({"audio": uniform(np.arange(10.0), sr=10)})
    with pytest.raises(DimensionError):
        tf.slice["mic", 0]


# ---------------------------------------------------------------------------
# RangeIndex size mismatch across a frame raises at construction
# ---------------------------------------------------------------------------


def test_range_index_size_mismatch_raises_at_construction():
    audio = uniform(np.zeros((8, 10)), sr=10, dims=("mic", "time"))
    mic_pos = wrap(np.zeros((3, 3)), dims=("mic", None))  # 3 != 8
    with pytest.raises(DimensionError):
        Frame({"audio": audio, "mic_pos": mic_pos})


def test_range_index_equal_sizes_accepted():
    audio = uniform(np.zeros((8, 10)), sr=10, dims=("mic", "time"))
    mic_pos = wrap(np.zeros((8, 3)), dims=("mic", None))
    tf = Frame({"audio": audio, "mic_pos": mic_pos})
    assert set(tf.keys()) == {"audio", "mic_pos"}


# ---------------------------------------------------------------------------
# LabelIndex dims with different sizes across leaves
# ---------------------------------------------------------------------------


def _label_frame() -> Frame:
    pos = wrap(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # (3, 2)
        dims=("mic", None),
        indexes={"mic": LabelIndex(("a", "b", "c"))},
    )
    gain = wrap(
        np.array([10.0, 20.0]),  # (2,)
        dims=("mic",),
        indexes={"mic": LabelIndex(("b", "d"))},
    )
    return Frame({"pos": pos, "gain": gain})


def test_label_dims_may_differ_in_size():
    tf = _label_frame()  # does not raise
    assert tf["pos"].dim_size("mic") == 3
    assert tf["gain"].dim_size("mic") == 2


def test_frame_sel_label_list_selects_per_leaf():
    tf = _label_frame()
    sub = tf.sel["mic", ["b", "d"]]
    # "pos" has only "b"; "gain" has both.
    assert sub["pos"].shape == (1, 2)
    np.testing.assert_array_equal(np.asarray(sub["pos"].data), [[3.0, 4.0]])
    pos_idx = sub["pos"].dim_index("mic")
    assert isinstance(pos_idx, LabelIndex)
    assert pos_idx.labels == ("b",)
    assert sub["gain"].shape == (2,)
    np.testing.assert_array_equal(np.asarray(sub["gain"].data), [10.0, 20.0])


def test_frame_sel_scalar_label_drops_dim():
    tf = _label_frame()
    sub = tf.sel["mic", "b"]
    assert sub["pos"].dims == (None,)
    np.testing.assert_array_equal(np.asarray(sub["pos"].data), [3.0, 4.0])
    assert sub["gain"].dims == ()
    assert float(sub["gain"].data) == 10.0


def test_frame_positional_slice_on_ragged_label_dim_raises():
    tf = _label_frame()  # sizes 3 vs 2 -> positional selection is ambiguous
    with pytest.raises(DimensionError):
        tf.slice["mic", 0]


def test_frame_sel_requires_label_index_on_all_occurrences():
    pos = wrap(
        np.array([1.0, 2.0, 3.0]),
        dims=("rotor",),
        indexes={"rotor": LabelIndex((1, 2, 3))},
    )
    tf = Frame({"pos": pos})
    with pytest.raises(DimensionError):
        tf.sel["mic", "b"]  # no leaf carries "mic"


# ---------------------------------------------------------------------------
# Mixing RangeIndex and LabelIndex on one dim name raises
# ---------------------------------------------------------------------------


def test_mixing_range_and_label_index_raises():
    labelled = wrap(
        np.array([1.0, 2.0, 3.0]),
        dims=("mic",),
        indexes={"mic": LabelIndex(("a", "b", "c"))},
    )
    plain = wrap(np.array([4.0, 5.0, 6.0]), dims=("mic",))  # default RangeIndex
    with pytest.raises(DimensionError):
        Frame({"labelled": labelled, "plain": plain})


def test_all_label_occurrences_accepted():
    a = wrap(
        np.array([1.0, 2.0, 3.0]),
        dims=("mic",),
        indexes={"mic": LabelIndex(("a", "b", "c"))},
    )
    b = wrap(
        np.array([4.0, 5.0]),
        dims=("mic",),
        indexes={"mic": LabelIndex(("a", "b"))},
    )
    tf = Frame({"a": a, "b": b})
    assert set(tf.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Dim ops leave Series lacking the dim untouched (including nested frames)
# ---------------------------------------------------------------------------


def test_frame_slice_recurses_into_nested_frame():
    inner = Frame(
        {
            "audio": uniform(
                np.arange(20, dtype=np.float64).reshape(2, 10), sr=10, dims=("mic", "time")
            )
        }
    )
    outer = Frame({"inner": inner, "mic_pos": wrap(np.zeros((2, 3)), dims=("mic", None))})
    sub = outer.slice["mic", 0]
    assert sub["inner"]["audio"].dims == ("time",)
    assert sub["inner"]["audio"].shape == (10,)
    assert sub["mic_pos"].shape == (3,)


def test_frame_slice_leaves_scalar_entries_untouched():
    tf = Frame(
        {
            "audio": uniform(np.zeros((2, 10)), sr=10, dims=("mic", "time")),
            "recording_id": "FLY124",
        }
    )
    sub = tf.slice["mic", 1]
    assert sub["recording_id"] == "FLY124"


# ---------------------------------------------------------------------------
# Series construction validation of dims
# ---------------------------------------------------------------------------


def test_duplicate_named_dims_raise():
    with pytest.raises(DimensionError):
        Series(np.zeros((2, 2)), ("mic", "mic"))


def test_index_key_must_be_named_dim():
    with pytest.raises(DimensionError):
        Series(np.zeros(3), (None,), {"mic": LabelIndex(("a", "b", "c"))})


def test_anonymous_axes_carry_no_index():
    s = wrap(np.zeros((2, 3)))
    assert s.dims == (None, None)
