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
    """Wraps a dataframe to run row-wise apply and groupby-apply in parallel.

    Parameters
    ----------
    df : pandas.DataFrame
        the dataframe to work on
    n_cores : int
        number of workers. -1 means the number of CPUs minus one, 0 means serial processing. It
        is capped by the number of rows.
    pbar : bool
        whether or not to show a progress bar for each worker
    parallelism : {'multiprocess', 'multithread'}
        whether to use processes (functions must be picklable) or threads

    Examples
    --------
    >>> import pandas as pd
    >>> from mt.pandas.parallel_apply import DataFrameParallel
    >>> df = pd.DataFrame({"a": [1, 2, 3, 4]})
    >>> dp = DataFrameParallel(df, 2, pbar=False, parallelism="multithread")
    >>> dp.apply(lambda row: row["a"] * 2, axis=1)
    0    2
    1    4
    2    6
    3    8
    dtype: int64
    >>> dp["a"].apply(lambda x: -x)
    0   -1
    1   -2
    2   -3
    3   -4
    Name: a, dtype: int64
    """
    def __init__(
        self, df: pd.DataFrame, n_cores: int, pbar: bool = True, parallelism: str = "multiprocess"
    ):
        self.df = df
        self.n_cores = n_cores
        self.pbar = pbar
        self.parallelism = parallelism

    # pylint: disable=unused-argument
    def apply(self, func, axis, raw: bool = False, result_type = None, args=(), **kwargs):
        """Wrapper on top of regular df.apply(fn), run in parallel over chunks of rows.

        Parameters
        ----------
        func : function
            function mapping each row (a :class:`pandas.Series`) to something
        axis : int
            must be 1 (rows), otherwise an AssertionError is raised
        raw, result_type, args
            ignored (kept for signature compatibility with :meth:`pandas.DataFrame.apply`)
        **kwargs
            passed to :meth:`pandas.DataFrame.apply`

        Returns
        -------
        pandas.Series or pandas.DataFrame
            the concatenated results of the chunks, in the original order
        """
        assert axis == 1, "Only axis=1 is supported in parallel df apply"
        return parallelize_dataframe(
            self.df, func, self.n_cores, self.pbar, self.parallelism, axis=axis, **kwargs
        )

    def groupby(self, *args, **kwargs):
        """Wrapper on top of regular df.groupby(col).

        All arguments are passed to :meth:`pandas.DataFrame.groupby`.

        Returns
        -------
        GroupByParallel
            an object whose `apply` runs in parallel, using the same `n_cores`, `pbar` and
            `parallelism` as this object.
        """
        return GroupByParallel(
            self.df.groupby(*args, **kwargs),
            self.n_cores,
            pbar=self.pbar,
            parallelism=self.parallelism,
        )

    def __getitem__(self, x):
        """Returns a :class:`SeriesParallel` wrapping column `x` with the same settings."""
        return SeriesParallel(self.df[x], self.n_cores, self.pbar, self.parallelism)

    def __str__(self) -> str:
        """Returns a header line followed by the string form of the dataframe."""
        f_str = f"[Parallel DataFrame - {self.n_cores} crores]\n" + self.df.__str__()
        return f_str
