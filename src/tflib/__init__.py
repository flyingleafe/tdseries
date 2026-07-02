"""tflib — immutable pytree frames of tensors with named, indexed dimensions.

Time is a first-class dimension indexed in exact int64 ticks
(``TICKS_PER_SECOND = 1e9``); slice / concat / shift compose exactly:
``frame.ticks[a:b] + frame.ticks[b:c] == frame.ticks[a:c]``.

Usage::

    import tflib as tf

    frame = tf.Frame({
        "audio":     tf.uniform(audio_8xT, sr=44100, dims=("mic", "time")),
        "mic_pos":   tf.wrap(pos_8x3, dims=("mic", None)),
        "rps":       tf.events(ts, rps_4xM, dims=("rotor", "time")),
        "rotor_pos": tf.wrap(rpos_4x3, dims=("rotor", None)),
        "vad":       tf.spans(starts, ends),
        "meta":      tf.Frame({"recording_id": "FLY124"}),
    })

    sub = frame.slice["mic", 0]      # audio (T,), mic_pos (3,) — same mic 0
    clip = frame.time[1.0:4.5]       # all temporal leaves cut, invariants kept
    one = frame["rps"]               # absolute-time Series; frame unchanged
"""

from ._ticks import TICKS_PER_SECOND, secs_to_ticks, ticks_to_secs
from .errors import DimensionError, DomainError, IncompatibleError
from .frame import Frame
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
from .series import Series, events, spans, uniform, wrap

__all__ = [
    "TICKS_PER_SECOND",
    "DimIndex",
    "DimensionError",
    "DomainError",
    "Frame",
    "GridIndex",
    "IncompatibleError",
    "LabelIndex",
    "RangeIndex",
    "SampleRate",
    "Series",
    "SpanIndex",
    "StampIndex",
    "TimeIndex",
    "events",
    "normalize_rate",
    "secs_to_ticks",
    "spans",
    "ticks_to_secs",
    "uniform",
    "wrap",
]
