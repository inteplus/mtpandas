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
    _range = tqdm(data) if pbar else data
    return [func(x) for x in _range]

def _get_multi_index_key(concat_res, key_data):
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

def _apply_on_groupby_parallel(df_grouped: DataFrameGroupBy, func: Callable, n_cores: int, pbar: bool,
                              keep_original_indexes: bool, parallelism: str) -> Union[pd.DataFrame, pd.Series]:
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
    """GroupByParallel implementation"""
    def __init__(self, df_grouped: DataFrameGroupBy, n_cores: int, pbar: bool = True,
                 keep_original_indexes: bool = False, parallelism: str = "multiprocess"):
        self.df_grouped = df_grouped
        self.n_cores = n_cores
        self.pbar = pbar
        self.keep_original_indexes = keep_original_indexes
        self.parallelism = parallelism

    def apply(self, func: Callable):
        """Wrapper on top of regular df.groupby(col).apply(fn)"""
        return _apply_on_groupby_parallel(self.df_grouped, func, self.n_cores, self.pbar,
                                          self.keep_original_indexes, self.parallelism)
