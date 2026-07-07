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
    """SeriesParallel implementation"""
    def __init__(self, series: pd.Series, n_cores: int, pbar: bool = True, parallelism: str = "multiprocess"):
        self.series = series
        self.n_cores = n_cores
        self.pbar = pbar
        self.parallelism = parallelism

    def apply(self, func: Callable) -> pd.Series:
        """Wrapper on top of regular ser.apply(fn)"""
        return parallelize_dataframe(self.series, func, self.n_cores, self.pbar, self.parallelism)
