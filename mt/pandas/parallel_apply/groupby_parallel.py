"""
Apply on df.groupby
Usage:
    standard => df.groupby([cols]).apply(f)
    parallel => DataFrameParallel(df, n_cores).groupby([cols]).apply(f)
             => GroupByPrallel(df.groupby[cols], n_cores).apply(f)
"""
from typing import Callable, List, Union
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.indexes.multi import MultiIndex
from tqdm import tqdm
import pandas as pd
import numpy as np
from .logger import logger
from .utils import get_n_cores


def _groupby_serial_func(data: List, func: Callable, pbar: bool = True) -> List:
    """Applies `func` serially on every dataframe of the list `data`, returning a list."""
    _range = tqdm(data) if pbar else data
    return [func(x) for x in _range]

def _get_multi_index_key(concat_res, key_data):
    """Builds a (group key, original index) multi-index from the per-group results."""
    new_key_data = []
    for i in range(len(concat_res)):
        # An awesome (KEY_1, 1), (KEY_2, 2), ..., (KEY_N, K) kind of deal
        n_keys = len(concat_res[i])
        multi_key = n_keys * [key_data[i]]
        counter_key = concat_res[i].index
        new_key = tuple(zip(multi_key, counter_key))
        new_key_data.extend(new_key)
    key_data = MultiIndex.from_tuples(new_key_data)
    return key_data

def _chunk_df(df_grouped: DataFrameGroupBy, n_cores: int) -> List[List]:
    """Splits the groups into chunks, returning the list of group keys and the list of chunks."""
    chunk_size = len(df_grouped) // n_cores + (len(df_grouped) % n_cores != 0)
    key_data = []
    chunked_data = []
    current_chunk = []
    for item in iter(df_grouped):
        key_data.append(item[0])
        current_chunk.append(item[1])
        if len(current_chunk) == chunk_size:
            chunked_data.append(current_chunk)
            current_chunk = []
    if 0 < len(current_chunk) < chunk_size:
        chunked_data.append(current_chunk)
    assert n_cores == len(chunked_data)
    return key_data, chunked_data

def _apply_on_groupby_parallel(df_grouped: DataFrameGroupBy, func: Callable, n_cores: int,
                              pbar: bool, keep_original_indexes: bool,
                              parallelism: str) -> Union[pd.DataFrame, pd.Series]:
    """Runs `func` on every group of `df_grouped` with a worker pool and concatenates results."""
    assert parallelism in ("multiprocess", "multithread"), parallelism
    n_cores = get_n_cores(n_cores, df_grouped)
    if n_cores == 0:
        logger.info("n_cores is set to 0, returning serial apply.")
        return df_grouped.apply(func)

    key_data, chunked_data = _chunk_df(df_grouped, n_cores)

    # Run the multi-process job
    pool = Pool(n_cores) if parallelism == "multiprocess" else ThreadPool(n_cores)
    pool_res = pool.map(partial(_groupby_serial_func, func=func, pbar=pbar), chunked_data)

    # Concatenate the result to preserve the original result of a regular groupby pandas code.
    concat_res = []
    for i in range(len(pool_res)):
        concat_res.extend(pool_res[i])

    if isinstance(concat_res[0], pd.Series):
        res = pd.DataFrame(concat_res)
    elif isinstance(concat_res[0], pd.DataFrame):
        res = pd.concat(concat_res)
        # SLOWS DOWN BY A LOT, LIKE 5x SERIAL TIME. TO INVESTIGATE
        if keep_original_indexes:
            key_data = _get_multi_index_key(concat_res, key_data)
        else:
            key_data = np.arange(len(res))
    else:
        res = pd.Series(concat_res)

    # Fix the index
    res.index = key_data
    res.index.name = df_grouped.keys
    return res


class GroupByParallel:
    """Wraps a pandas groupby object to run apply in parallel.

    The groups are split into `n_cores` chunks of consecutive groups, each chunk being processed
    by one worker.

    Parameters
    ----------
    df_grouped : pandas.core.groupby.DataFrameGroupBy
        the grouped dataframe, e.g. ``df.groupby("col")``
    n_cores : int
        number of workers. -1 means the number of CPUs minus one, 0 means serial processing. It
        is capped by the number of groups.
    pbar : bool
        whether or not to show a progress bar for each worker
    keep_original_indexes : bool
        when `func` returns dataframes, whether or not to build a (group key, original index)
        multi-index for the output. Otherwise a fresh integer index is used. Slow if True.
    parallelism : {'multiprocess', 'multithread'}
        whether to use processes (functions must be picklable) or threads

    Examples
    --------
    >>> import pandas as pd
    >>> from mt.pandas.parallel_apply import GroupByParallel
    >>> df = pd.DataFrame({"g": [0, 0, 1, 1], "a": [1, 2, 3, 4]})
    >>> def total(d):
    ...     return d["a"].sum()
    >>> gp = GroupByParallel(df.groupby("g"), 2, pbar=False, parallelism="multithread")
    >>> gp.apply(total)
    g
    0    3
    1    7
    dtype: int64
    """
    def __init__(self, df_grouped: DataFrameGroupBy, n_cores: int, pbar: bool = True,
                 keep_original_indexes: bool = False, parallelism: str = "multiprocess"):
        self.df_grouped = df_grouped
        self.n_cores = n_cores
        self.pbar = pbar
        self.keep_original_indexes = keep_original_indexes
        self.parallelism = parallelism

    def apply(self, func: Callable):
        """Wrapper on top of regular df.groupby(col).apply(fn), run in parallel over the groups.

        Parameters
        ----------
        func : function
            function taking the dataframe of one group and returning a dataframe, a series or a
            scalar

        Returns
        -------
        pandas.DataFrame or pandas.Series
            a dataframe if `func` returns series or dataframes, otherwise a series. The index is
            the group keys (series results) or follows `keep_original_indexes` (dataframe
            results).
        """
        return _apply_on_groupby_parallel(self.df_grouped, func, self.n_cores, self.pbar,
                                          self.keep_original_indexes, self.parallelism)
