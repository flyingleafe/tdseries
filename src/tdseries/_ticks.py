"""Fixed-point time conversion helpers.

All time inside the library is stored as int64 tick counts at a fixed
TICKS_PER_SECOND (nanoseconds).  This module is the conversion boundary
between the float-seconds ecosystem and the library's exact int64 storage.
"""

from __future__ import annotations

import numpy as np

TICKS_PER_SECOND: int = 1_000_000_000


# ---- scalar ------------------------------------------------------------


def secs_to_ticks(seconds: float) -> int:
    """Convert scalar float-seconds to the nearest int64 tick."""
    return round(seconds * TICKS_PER_SECOND)


def ticks_to_secs(ticks: int) -> float:
    """Convert scalar int64 ticks to float seconds."""
    return ticks / TICKS_PER_SECOND


# ---- arrays ------------------------------------------------------------


def secs_array_to_ticks(seconds: np.ndarray) -> np.ndarray:
    """Convert an array of float seconds to int64 ticks (nearest)."""
    result = np.rint(np.asarray(seconds, dtype=np.float64) * TICKS_PER_SECOND)
    return result.astype(np.int64)


def ticks_array_to_secs(ticks: np.ndarray) -> np.ndarray:
    """Convert an array of int64 ticks to float seconds."""
    return np.asarray(ticks, dtype=np.float64) / TICKS_PER_SECOND


# ---- coercion ----------------------------------------------------------


def to_ticks(value: float | int) -> int:
    """Coerce a ``float`` (seconds) or ``int`` (ticks) to int ticks.

    The single conversion point for slice / shift / constructor arguments:
    float seconds are quantised once and never re-rounded.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    return secs_to_ticks(float(value))


def seconds_to_ticks_exact(value: float | int) -> int:
    """Coerce a *seconds* value to ticks.  Unlike :func:`to_ticks`, an int
    here means whole seconds (exact multiplication), not ticks.  Used by the
    ``.time[...]`` accessor where everything is in seconds.
    """
    if isinstance(value, (int, np.integer)):
        return int(value) * TICKS_PER_SECOND
    return secs_to_ticks(float(value))
