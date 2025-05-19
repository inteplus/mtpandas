"""Additional utilities dealing with dataframes."""

import asyncio
import pandas as pd
from tqdm.auto import tqdm
from pandas_parallel_apply import DataFrameParallel

from mt import tp, logg, ctx
from mt.base import LogicError


__all__ = [
    "rename_column",
    "row_apply",
    "row_transform_asyn",
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


async def row_transform_asyn(
    df: pd.DataFrame,
    func,
    func_args: tuple = (),
    func_kwargs: dict = {},
    max_concurrency: int = 1,
    bar_unit="it",
    context_vars: dict = {},
) -> pd.DataFrame:
    """Transforms each row of a :class:`pandas.DataFrame` to another row, using an asyn function, and optionally with a progress bar.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    func : function
        an asyn function to map each row of the dataframe to something. Its first positional
        argument represents the input row. It must return a :class:`pandas.Series` as output.
    func_args : tuple, optional
        additional positional arguments to be passed to the function
    func_kwargs : dict, optional
        additional keyword arguments to be passed to the function
    max_concurrency : int
        maximum number of concurrent rows to process at a time. If a number greater than 1 is
        provided, the processing of each row is then converted into an asyncio task to be run
        concurrently.
    bar_unit : str, optional
        unit name to be passed to the progress bar. If None is provided, no bar is displayed.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    pandas.DataFrame
        output dataframe
    """

    if bar_unit is not None:
        bar = tqdm(total=len(df), unit=bar_unit)

        async def func2(row, *args, context_vars: dict = {}, **kwargs):
            res = await func(row, *args, context_vars=context_vars, **kwargs)
            bar.update()
            return res

        with bar:
            return await row_transform_asyn(
                df,
                func2,
                func_args=func_args,
                func_kwargs=func_kwargs,
                max_concurrency=max_concurrency,
                bar_unit=None,
                context_vars=context_vars,
            )

    N = len(df)
    if N == 0:
        raise ValueError("Cannot process an empty dataframe.")

    if (N <= max_concurrency) or (max_concurrency == 1):  # too few or sequential
        l_records = []
        for idx, row in df.iterrows():
            try:
                out_row = await func(
                    row, *func_args, context_vars=context_vars, **func_kwargs
                )
                l_records.append((idx, out_row))
            except Exception as e:
                raise LogicError(
                    "Row transformation has encountered an exception.",
                    debug={"idx": idx, "row": row},
                    causing_error=e,
                )
    else:
        i = 0
        l_records = []
        s_tasks = set()

        while i < N or len(s_tasks) > 0:
            # push
            pushed = False
            while i < N and len(s_tasks) < max_concurrency:
                coro = func(
                    df.iloc[i], *func_args, context_vars=context_vars, **func_kwargs
                )
                task = asyncio.create_task(coro, name=str(i))
                s_tasks.add(task)
                i += 1
                pushed = True

            # wait a bit
            await asyncio.sleep(0.1)
            if pushed:
                sleep_cnt = 0
            else:
                sleep_cnt += 1
            if sleep_cnt >= 1800:
                loop = asyncio.get_running_loop()
                loop.set_debug(True)
            if sleep_cnt >= 3000:
                debug = {
                    "N": N,
                    "i": i,
                    "s_tasks": [task.get_name() for task in s_tasks],
                }
                raise LogicError("No task has been done for 5 minutes.", debug=debug)

            # get the status of each event
            s_pending = set()
            s_done = set()
            s_error = set()
            s_cancelled = set()
            for task in s_tasks:
                if not task.done():
                    s_pending.add(task)
                elif task.cancelled():
                    s_cancelled.add(task)
                elif task.exception() is not None:
                    s_error.add(task)
                else:
                    s_done.add(task)

            # raise a common LogicError for all cancelled tasks
            if s_cancelled:
                rows = [int(task.get_name()) for task in s_cancelled]
                raise LogicError(
                    "Cancelled row transformations detected.",
                    debug={"rows": df.iloc[rows]},
                )

            # raise a common LogicError for all tasks that have generated an exception
            if s_error:
                rows = [int(task.get_name()) for task in s_error]
                for task in s_error:
                    break
                raise LogicError(
                    "Exceptions raised in some row transformations. First exception reported here.",
                    debug={"rows": df.iloc[rows]},
                    causing_error=task.exception(),
                )

            # process done tasks
            if s_done:
                sleep_cnt = 0
                for task in s_done:
                    j = int(task.get_name())
                    rec = (df.index[j], task.result())
                    l_records.append(rec)

            # update s_tasks
            s_tasks = s_pending

    index, data = zip(*l_records)
    df2 = pd.DataFrame(index=index, data=data)
    df2.index.name = df.index.name
    return df2


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
