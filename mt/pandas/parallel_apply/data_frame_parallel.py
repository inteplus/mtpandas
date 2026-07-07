"""
Wrapper on top of pd.DataFrame with some parallel operations
Apply on column
Usage:
    standard => df[col_name].apply(f)
    parallel => DataFrameParallel(df, n_cores)[col_name].apply(f)

Apply on row
Usage:
    standard => df.apply(f, axis=1)
    parallel => DataFrameParallel(df, n_cores).apply(f, axis=1)
"""
import pandas as pd

from .series_parallel import SeriesParallel
from .groupby_parallel import GroupByParallel
from .utils import parallelize_dataframe

class DataFrameParallel:
    """DataFrameParallel implementation"""
    def __init__(self, df: pd.DataFrame, n_cores: int, pbar: bool = True, parallelism: str = "multiprocess"):
        self.df = df
        self.n_cores = n_cores
        self.pbar = pbar
        self.parallelism = parallelism

    # pylint: disable=unused-argument
    def apply(self, func, axis, raw: bool = False, result_type = None, args=(), **kwargs):
        """Wrapper on top of regular df.apply(fn)"""
        assert axis == 1, "Only axis=1 is supported in parallel df apply"
        return parallelize_dataframe(self.df, func, self.n_cores, self.pbar, self.parallelism, axis=axis, **kwargs)

    def groupby(self, *args, **kwargs):
        """Wrapper on top of regular df.groupby(col)"""
        return GroupByParallel(self.df.groupby(*args, **kwargs), self.n_cores, self.pbar, self.parallelism)

    def __getitem__(self, x):
        return SeriesParallel(self.df[x], self.n_cores, self.pbar, self.parallelism)

    def __str__(self) -> str:
        f_str = f"[Parallel DataFrame - {self.n_cores} crores]\n" + self.df.__str__()
        return f_str
