"""Custom accessors for pandas. Placeholder to load other modules."""

from .json import JsonAccessor
from .ndarray import NdarrayAccessor
from .word import WordAccessor

import pandas as pd

_PISO_TARGETS = (pd.IntervalIndex, pd.arrays.IntervalArray)


class _LazyPisoAccessor:
    """Placeholder for the `piso` accessor that imports piso only on first use.

    piso is an optional dependency, installed via ``mtpandas[interval]``. On first access of
    ``.piso`` the placeholders are removed, piso's own accessors are registered, and the access is
    redone.
    """

    def __get__(self, obj, cls):
        for target in _PISO_TARGETS:
            if isinstance(target.__dict__.get("piso"), _LazyPisoAccessor):
                delattr(target, "piso")
        try:
            import piso
        except ImportError as e:
            for target in _PISO_TARGETS:  # restore placeholders
                setattr(target, "piso", _LazyPisoAccessor())
            raise AttributeError(
                "The 'piso' accessor requires the optional dependency piso. "
                "Install it with 'pip install mtpandas[interval]'."
            ) from e
        piso.register_accessors()
        return getattr(cls if obj is None else obj, "piso")


for _target in _PISO_TARGETS:
    if not hasattr(_target, "piso"):
        setattr(_target, "piso", _LazyPisoAccessor())
