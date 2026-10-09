"""Vendored copy of the `pandas_parallel_apply` package.

Brought in-tree (instead of depending on the PyPI package) so we can
monkey-patch it. Source: https://gitlab.com/meehai/pandas-parallel-apply

Three wrappers mirror the pandas apply APIs and run them over several cores, either with
multiprocessing (``parallelism="multiprocess"``, default) or with threads
(``parallelism="multithread"``): :class:`SeriesParallel` for ``Series.apply``,
:class:`DataFrameParallel` for ``DataFrame.apply(axis=1)`` and :class:`GroupByParallel` for
``DataFrame.groupby(...).apply``. For multiprocessing the function must be picklable.
"""
from .data_frame_parallel import DataFrameParallel
from .series_parallel import SeriesParallel
from .groupby_parallel import GroupByParallel
