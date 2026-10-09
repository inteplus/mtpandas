"""General module for parallelizing a dataframe apply function on a column (series) or entire row"""

from typing import Callable, Union
from multiprocessing import cpu_count, current_process
from multiprocessing.pool import Pool, ThreadPool
from threading import current_thread
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm

from .logger import logger

def get_n_cores(n_cores: int, df: Union[pd.DataFrame, pd.Series]) -> int:
    """
    Returns the actual n_cores used for the parallel operation. cpu_count() represents the total
    amount of phyisical cores. Possible cases:
    - n_cores < -1: will throw an AssertionError
    - n_cores = -1: will return cpu_count() - 1
    - n_cores in [0, cpu_count()]: will return the number as is
    - n_cores > cpu_count(): will return the number as is, but will throw a warning

    In every case the result is capped by ``len(df)`` (with a warning). A result of 0 means that
    the caller should process serially.

    Parameters
    ----------
    n_cores : int
        requested number of cores
    df : pandas.DataFrame or pandas.Series
        the object to be processed

    Returns
    -------
    int
        the number of cores to use

    Examples
    --------
    >>> import pandas as pd
    >>> from mt.pandas.parallel_apply.utils import get_n_cores
    >>> get_n_cores(0, pd.Series([1, 2, 3]))
    0
    >>> get_n_cores(2, pd.Series([1, 2, 3]))
    2
    """
    assert n_cores >= -1, f"n_cores cannot be negative, except -1. Got {n_cores}"
    if n_cores == -1:
        n_cores = cpu_count() - 1
        logger.debug(
            f"n_cores -1 was provided. Using total number of physical cores - 1: {n_cores}"
        )
    if n_cores > cpu_count():
        logger.warning(
            f"n_cores is greater than the number of physical cores: {n_cores} vs {cpu_count()}"
        )
    if n_cores > len(df):
        logger.warning(
            f"n_cores is greater than the length of the df, return that: {n_cores} vs {len(df)}"
        )
        n_cores = len(df)
    return n_cores

# pylint: disable=protected-access
def _worker_fn(df: pd.DataFrame, fn: Callable, pbar: bool, kwargs: dict):
    """worker_fn: calls df.progress_apply or df.apply. For pbar, will use position based on
    thread/process index"""
    if not pbar:
        return df.apply(fn, **kwargs)

    if len(current_process()._identity) == 0:
        position = int(current_thread().name.split(" ")[0].split("Thread-")[1])
        desc = f"Thread-{position}"
    else:
        position = current_process()._identity[0] - 1
        desc = f"Process-{position}"

    tqdm.pandas(position=position, desc=desc)
    return df.progress_apply(fn, **kwargs)

def parallelize_dataframe(df: Union[pd.DataFrame, pd.Series], func: Callable, n_cores: int,
                          pbar: bool, parallelism: str, **kwargs) -> pd.DataFrame:
    """Function used to split a dataframe in n sub dataframes, based on the number of cores we
    want to use.

    Each chunk is processed by ``apply`` in a worker and the results are concatenated in order.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        the object to split row-wise
    func : function
        the function to apply
    n_cores : int
        requested number of cores, see :func:`get_n_cores`. With 0, a plain ``df.apply`` is run.
    pbar : bool
        whether or not to show a progress bar per worker
    parallelism : {'multiprocess', 'multithread'}
        pool type
    **kwargs
        passed to ``apply`` (for a dataframe, e.g. ``axis=1``)

    Returns
    -------
    pandas.DataFrame or pandas.Series
        the concatenated results

    Examples
    --------
    >>> import pandas as pd
    >>> from mt.pandas.parallel_apply.utils import parallelize_dataframe
    >>> parallelize_dataframe(pd.Series([1, 2, 3]), lambda x: x * 3, 2, False, "multithread")
    0    3
    1    6
    2    9
    dtype: int64
    """
    assert parallelism in ("multiprocess", "multithread"), parallelism
    n_cores = get_n_cores(n_cores, df)
    if n_cores == 0:
        logger.debug("n_cores is set to 0, returning serial function")
        return df.apply(func, **kwargs)
    logger.debug(f"Parallelizing apply on df (rows: {len(df)}) with {n_cores} cores")

    # np.array_split(df, n_cores) would coerce the DataFrame/Series into a plain
    # ndarray on newer numpy releases, so split via row-index arrays instead.
    idx_split = np.array_split(np.arange(len(df)), n_cores)
    df_split = [df.iloc[idx] for idx in idx_split]
    pool_fn = partial(_worker_fn, fn=func, pbar=pbar, kwargs=kwargs)
    pool = Pool(n_cores) if parallelism == "multiprocess" else ThreadPool(n_cores)
    pool_res = pool.map(pool_fn, df_split)

    # This should use less memory than pd.concat(pool_res)
    final_df = pool_res[0]
    for res_df in pool_res[1: ]:
        final_df = pd.concat([final_df, res_df], copy=False)
        del res_df
    return final_df
