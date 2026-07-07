"""Vendored copy of the `pandas_parallel_apply` package.

Brought in-tree (instead of depending on the PyPI package) so we can
monkey-patch it. Source: https://gitlab.com/meehai/pandas-parallel-apply
"""
from .data_frame_parallel import DataFrameParallel
from .series_parallel import SeriesParallel
from .groupby_parallel import GroupByParallel
