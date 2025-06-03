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
    func_kwds: dict = {},
    max_concurrency: int = 1,
    bar_unit="it",
    timeout: tp.Optional[float] = None,
    context_vars: dict = {},
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
) -> pd.DataFrame:
    """Transforms each row of a :class:`pandas.DataFrame` to another row, using an asyn function, and optionally with a progress bar.

    The order of the rows being transformed is not preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    func : function
        an asyn function to map each row of the dataframe to something. Its first positional
        argument represents the input row. It must return a :class:`pandas.Series` as output.
    func_args : tuple, optional
        additional positional arguments to be passed to the function
    func_kwds : dict, optional
        additional keyword arguments to be passed to the function
    max_concurrency : int
        maximum number of concurrent rows to process at a time. If a number greater than 1 is
        provided, the processing of each row is then converted into an asyncio task to be run
        concurrently.
    bar_unit : str, optional
        unit name to be passed to the progress bar. If None is provided, no bar is displayed.
    timeout : float, optional
        maximum number of seconds for each row to be transformed, before a TimeoutError exception
        is raised. Value 300, representing 5 minutes, is reasonable.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes.

    Returns
    -------
    pandas.DataFrame
        output dataframe. Apart from the columns provided by argument `func`, there is a column
        called 'row_transform_exception' that contains any exception encountered when processing a
        row.
    """

    if bar_unit is not None:
        bar = tqdm(total=len(df), unit=bar_unit)

        async def func2(row, *args, context_vars: dict = {}, **kwds):
            res = await func(row, *args, context_vars=context_vars, **kwds)
            bar.update()
            return res

        with bar:
            return await row_transform_asyn(
                df,
                func2,
                func_args=func_args,
                func_kwds=func_kwds,
                max_concurrency=max_concurrency,
                bar_unit=None,
                timeout=timeout,
                context_vars=context_vars,
                logger=logger,
            )

    N = len(df)
    if N == 0:
        raise ValueError("Cannot process an empty dataframe.")

    def error_series(e: Exception) -> pd.Series:
        return pd.Series(data={"row_transform_exception": e})

    if (N <= max_concurrency) or (max_concurrency == 1):  # too few or sequential
        l_records = []
        for idx, row in df.iterrows():
            try:
                out_row = await func(
                    row, *func_args, context_vars=context_vars, **func_kwds
                )
                l_records.append((idx, out_row))
            except Exception as e:
                if logger:
                    logger.warn_last_exception()
                l_records.append((idx, error_series(e)))
    else:
        if timeout is not None:

            async def func2(row, *args, context_vars: dict = {}, **kwds):
                async with asyncio.timeout(timeout):
                    return await func(row, *args, context_vars=context_vars, **kwds)

        else:
            func2 = func

        i = 0
        l_records = []
        d_tasks = {}

        while i < N or len(d_tasks) > 0:
            # push
            pushed = False
            while i < N and len(d_tasks) < max_concurrency:
                coro = func2(
                    df.iloc[i], *func_args, context_vars=context_vars, **func_kwds
                )
                task = asyncio.create_task(coro)
                d_tasks[task] = i
                i += 1
                pushed = True

            # wait a bit
            await asyncio.sleep(0.1)
            if pushed:
                sleep_cnt = 0
            else:
                sleep_cnt += 1
            if timeout is not None and sleep_cnt >= timeout * 10:
                if logger:
                    rows = list(d_tasks.values())
                    logger.warn(f"Timed out transforming rows:\n{df.iloc[rows]}")

                for task, j in d_tasks.items():
                    e = TimeoutError("Timeout while concurrently transforming the row.")
                    l_records.append((df.index[j], error_series(e)))
                    task.cancel()

                d_tasks = {}

            # get the status of each event
            d_pending = {}
            d_done = {}
            d_error = {}
            d_cancelled = {}
            for task, j in d_tasks.items():
                if not task.done():
                    d_pending[task] = j
                elif task.cancelled():
                    d_cancelled[task] = j
                elif task.exception() is not None:
                    d_error[task] = j
                else:
                    d_done[task] = j

            # unexpectedly cancelled tasks
            if d_cancelled:
                if logger:
                    rows = list(d_cancelled.values())
                    logger.warn(
                        f"Transformation unexpectedly cancelled for rows:\n{df.iloc[rows]}"
                    )

                for task, j in d_cancelled.items():
                    e = asyncio.CancelledError(
                        "Concurrent task cancelled unexpectedly."
                    )
                    l_records.append((df.index[j], error_series(e)))

            # tasks with raised exceptions
            if d_error:
                if logger:
                    rows = list(d_error.values())
                    logger.warn(
                        f"Exceptions raised transforming rows:\n{df.iloc[rows]}"
                    )

                e = None
                for task, j in d_error.items():
                    if logger and (e is None):
                        e = task.exception()
                        with logger.scoped_warn("First exception"):
                            logger.warn_exception(e, row=df.iloc[j])
                    else:
                        e = task.exception()
                    l_records.append((df.index[j], error_series(e)))

            # done tasks
            if d_done:
                sleep_cnt = 0
                for task, j in d_done.items():
                    out_row = task.result()
                    if not isinstance(out_row, pd.Series):
                        msg = f"Transformed row {j} is not a series: {out_row}."
                        logg.warn(msg, logger=logger)
                    l_records.append((df.index[j], out_row))

            # update d_tasks
            d_tasks = d_pending

    index, data = zip(*l_records)
    try:
        df2 = pd.DataFrame(index=index, data=data)
        df2.index.name = df.index.name
    except Exception as e:
        debug = {
            "len": len(index),
            "index[:10]": index[:10],
            "data[:10]": data[:10],
        }
        raise LogicError(
            "Error detected while putting together the output dataframe.",
            debug=debug,
            causing_error=e,
        )
    if "row_transform_exception" not in df2.columns:
        df2["row_transform_exception"] = None
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
