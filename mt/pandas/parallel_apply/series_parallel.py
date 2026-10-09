"""
Apply on series
Usage:
    standard => series.apply(f)
    parallel => SeriesParallel(series, n_cores).apply(f)
             => SeriesParallel(df[col], n_cores).apply(f)
"""
from typing import Callable
import pandas as pd

from .utils import parallelize_dataframe

class SeriesParallel:
    """Wraps a series to run apply in parallel.

    Parameters
    ----------
    series : pandas.Series
        the series to work on
    n_cores : int
        number of workers. -1 means the number of CPUs minus one, 0 means serial processing. It
        is capped by the length of the series.
    pbar : bool
        whether or not to show a progress bar for each worker
    parallelism : {'multiprocess', 'multithread'}
        whether to use processes (functions must be picklable) or threads

    Examples
    --------
    >>> import pandas as pd
    >>> from mt.pandas.parallel_apply import SeriesParallel
    >>> sp = SeriesParallel(pd.Series([1, 2, 3]), 2, pbar=False, parallelism="multithread")
    >>> sp.apply(lambda x: x + 1)
    0    2
    1    3
    2    4
    dtype: int64
    """
    def __init__(
        self, series: pd.Series, n_cores: int, pbar: bool = True, parallelism: str = "multiprocess"
    ):
        self.series = series
        self.n_cores = n_cores
        self.pbar = pbar
        self.parallelism = parallelism

    def apply(self, func: Callable) -> pd.Series:
        """Wrapper on top of regular ser.apply(fn), run in parallel over chunks of the series.

        Parameters
        ----------
        func : function
            function mapping each item to something

        Returns
        -------
        pandas.Series
            the concatenated results of the chunks, in the original order
        """
        return parallelize_dataframe(self.series, func, self.n_cores, self.pbar, self.parallelism)
