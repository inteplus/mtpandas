"""Additional utilities dealing with dataframes."""


import pandas as pd
from tqdm.auto import tqdm
from pandas_parallel_apply import DataFrameParallel

from mt import tp, logg, ctx


__all__ = [
    "rename_column",
    "row_apply",
    "parallel_apply",
    "warn_duplicate_records",
    "filter_rows",
]


def rename_column(df: pd.DataFrame, old_column: str, new_column: str) -> bool:
    """Renames a column in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        the dataframe to work on
    old_column : str
        the column name to be renamed
    new_column : str
        the new column name

    Returns
    -------
    bool
        whether or not the column has been renamed
    """
    if old_column not in df.columns:
        return False

    columns = list(df.columns)
    columns[columns.index(old_column)] = new_column
    df.columns = columns
    return True


def row_apply(df: pd.DataFrame, func, bar_unit="it") -> pd.DataFrame:
    """Applies a function on every row of a pandas.DataFrame, optionally with a progress bar.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    func : function
        a function to map each row of the dataframe to something
    bar_unit : str, optional
        unit name to be passed to the progress bar. If None is provided, no bar is displayed.

    Returns
    -------
    pandas.DataFrame
        output series by invoking `df.apply`. And a progress bar is shown if asked.
    """

    if bar_unit is None:
        return df.apply(func, axis=1)

    bar = tqdm(total=len(df), unit=bar_unit)

    def func2(row):
        res = func(row)
        bar.update()
        return res

    with bar:
        return df.apply(func2, axis=1)


def parallel_apply(
    df: pd.DataFrame,
    func,
    axis: int = 1,
    n_cores: int = -1,
    parallelism: str = "multiprocess",
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
    scoped_msg: tp.Optional[str] = None,
) -> pd.Series:
    """Parallel-applies a function on every row or column of a pandas.DataFrame, optionally with a progress bar.

    The method wraps class:`pandas_parallel_apply.DataFrameParallel`. The default axis is on rows.
    The progress bars are shown if and only if a logger is provided.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    func : function
        a function to map a series to a series. It must be pickable for parallel processing.
    axis : {0,1}
        axis of applying. 1 for rows (default). 0 for columns.
    n_cores : int
        number of CPUs to use. Passed as-is to :class:`pandas_parallel_apply.DataFrameParallel`.
    parallelism : {'multithread', 'multiprocess'}
        multi-threading or multi-processing. Passed as-is to
        :class:`pandas_parallel_apply.DataFrameParallel`.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes.
    scoped_msg : str, optional
        whether or not to scoped_info the progress bars. Only valid if a logger is provided

    Returns
    -------
    pandas.DataFrame
        output dataframe by invoking `df.apply`.

    See Also
    --------
    pandas_parallel_apply.DataFrameParallel
        the wrapped class for the parallel_apply purpose
    """

    if logger:
        dp = DataFrameParallel(df, n_cores=n_cores, parallelism=parallelism, pbar=True)
        if scoped_msg:
            context = logger.scoped_info(scoped_msg)
        else:
            context = ctx.nullcontext()
    else:
        dp = DataFrameParallel(df, n_cores=n_cores, parallelism=parallelism, pbar=False)
        context = ctx.nullcontext()

    with context:
        return dp.apply(func, axis)


def warn_duplicate_records(
    df: pd.DataFrame,
    keys: list,
    msg_format: str = "Detected {dup_cnt}/{rec_cnt} duplicate records.",
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
):
    """Warns of duplicate records in the dataframe based on a list of keys.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    keys : list
        list of column names
    msg_format : str, optional
        the message to be logged. Two keyword arguments will be provided 'rec_cnt' and 'dup_cnt'.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes.
    """
    if not logger:
        return

    cnt0 = len(df)
    if not isinstance(keys, list):
        keys = [keys]
    cnt1 = len(df[keys].drop_duplicates())
    if cnt1 < cnt0:
        logger.warning(msg_format.format(dup_cnt=cnt0 - cnt1, rec_cnt=cnt0))


def filter_rows(
    df: pd.DataFrame,
    s: pd.Series,
    msg_format: str = None,
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
) -> pd.DataFrame:
    """Returns `df[s]` but warn if the number of rows drops.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    s : pandas.Series
        the boolean series to filter the rows of `df`. Must be of the same size as `df`.
    msg_format : str, optional
        the message to be logged. Two keyword arguments will be provided 'n_before' and 'n_after'.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes.
    """

    n_before = len(df)
    if n_before == 0:
        return df

    df2 = df[s]
    n_after = len(df2)

    if n_after == n_before:
        return df2

    if msg_format is None:
        msg_format = "After filtering, the number of rows has reduced from {n_before} to {n_after}."
    msg = msg_format.format(n_before=n_before, n_after=n_after)
    logg.warn(msg, logger=logger)

    return df2
