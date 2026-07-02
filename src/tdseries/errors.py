"""Library exceptions."""

from __future__ import annotations


class DomainError(ValueError):
    """A slice request lies outside the declared domain."""


class IncompatibleError(ValueError):
    """Two objects cannot be concatenated or merged (mismatched sample
    rate, dims, index kind, conflicting invariant entries, ...)."""


class DimensionError(ValueError):
    """A named-dimension constraint is violated (unknown dim, size
    mismatch across a frame, positional slicing of a ragged dim, ...)."""
