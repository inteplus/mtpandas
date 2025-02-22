"""Extra functions to augment pandas.Series."""

import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from pandas_parallel_apply import SeriesParallel

from mt import tp, logg, ctx


__all__ = [
    "Series4json",
    "json4Series",
    "to_categorical",
    "series_apply",
    "series_parallel_apply",
    "stats",
    "int_list",
]


def Series4json(obj):
    """Converts a json-like object in to a pandas.Series.

    Parameters
    ----------
    obj : json_like
        a json-like object

    Returns
    -------
    pandas.Series
        output Series

    Notes
    -----
    A json-like object contains 2 array_likes, each has length K. The first array represents the index array. The second array represents the value array.
    """
    return pd.Series(index=obj[0], data=obj[1])


def json4Series(obj):
    """Converts a pandas.Series into a json-like object.

    Parameters
    ----------
    obj : pandas.Series
        input Series

    Returns
    -------
    json_like
        a json-like object

    Notes
    -----
    A json-like object contains 2 array_likes, each has length K. The first array represents the index array. The second array represents the value array.
    """
    return [obj.index.tolist(), obj.tolist()]


def to_categorical(series, value_list, missing_values="raise_exception", logger=None):
    """Converts a string series representing a categorical field into a zero-indexed categorical field and a one-hot field in json format.

    Parameters
    ----------
    series : pandas.Series
        an indexed series whose values are strings representing categories
    value_list : list
        list of accepted values to be converted into integers. The first value is converted into 0, the second value into 2, and so on. The length of the list is the number of categories.
    missing_values : {'raise_exception', 'remove_and_warn', 'remove_in_silence'}
        policy for treating missing values upon encountered. If 'raise_exception' is specified, raise a ValueError when encountering the first missing value. If 'remove_and_warn' is specified, all missing values are warned to the logger, but records containing missing values are removed from output. If 'remove_in_silence' is specified, all records containing missing values are removed from output in slience.
    logger : logging.Logger to equivalent, optional
        logger for debugging purposes

    Returns
    -------
    df : `pandas.DataFrame(columns=['cat_id', 'one_hot'])`
        indexed dataframe with the same index as `series`. However, some records may be removed due to containing missing values. Field `cat_id` contains the converted zero-indexed categorical id intenger. Field `one_hot` contains a numpy 1d array representing the one-hot representation of the categorical id. The number of components of the one-hot vector is the same as the length of `value_list`.

    Raises
    ------
    ValueError
        if something has gone wrong
    """
    df = series.to_frame("cat")  # to dataframe

    # check for missing values
    df2 = df[~df["cat"].isin(value_list)]
    if len(df2) > 0:
        missing_list = df2["cat"].drop_duplicates().tolist()
        if missing_values == "raise_exception":
            raise ValueError("Missing values detected: {}".format(missing_list))
        elif missing_values == "remove_and_warn":
            df = df[df["cat"].isin(value_list)]
            if logger:
                logger.warn(
                    "Missing values detected and removed: {}".format(missing_list)
                )
        elif missing_values == "remove_in_silence":
            df = df[df["cat"].isin(value_list)]
        else:
            raise ValueError(
                "Unknown policy to deal with missing values: {}. Accepted policies are 'raise_exception', 'remove_and_warn' or 'remove_in_silence'.".format(
                    missing_values
                )
            )

    # cat id
    df["cat_id"] = df["cat"].map(value_list.index)

    # one hot
    eye = np.eye(len(value_list))
    eye_list = {i: eye[i] for i in range(len(value_list))}
    df["one_hot"] = df["cat_id"].map(eye_list)

    return df.drop("cat", axis=1)


def series_apply(
    s: pd.Series,
    func,
    bar_unit="it",
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
    func_uses_logger: bool = False,
) -> pd.Series:
    """Applies a function on every item of a pandas.Series, optionally with a progress bar.

    Parameters
    ----------
    s : pandas.Series
        a series
    func : function
        a function to map each item of the series to something
    bar_unit : str, optional
        unit name to be passed to the progress bar. If None is provided, no bar is displayed.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes. Only used if `bar_unit` is not None.
    func_uses_logger : bool
        whether or not the function takes `logger` as an additional keyword argument. Valid only if
        `logger` is used.

    Returns
    -------
    pandas.Series
        output series by invoking `s.apply`. And a progress bar is shown if asked.
    """

    if bar_unit is None:
        return s.apply(func)

    bar = tqdm(total=len(s), unit=bar_unit)

    def func2(x):
        try:
            if func_uses_logger:
                res = func(x, logger=logger)
            else:
                res = func(x)
        except:
            if logger:
                logger.error(
                    "Caught the below exception while series_applying a sequence:"
                )
                logger.error(f"  Item: {x}")
            raise
        bar.update()
        return res

    with bar:
        return s.apply(func2)


def series_parallel_apply(
    s: pd.Series,
    func,
    n_cores: int = -1,
    parallelism: str = "multiprocess",
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
    scoped_msg: tp.Optional[str] = None,
) -> pd.Series:
    """Parallel-applies a function on every item of a pandas.Series, optionally with a progress bar.

    The method wraps class:`pandas_parallel_apply.SeriesParallel`. The progress bars are shown if
    and only if a logger is provided.

    Parameters
    ----------
    s : pandas.Series
        a series
    func : function
        a function to map each item of the series to something. It must be pickable for parallel
        processing.
    n_cores : int
        number of CPUs to use. Passed as-is to :class:`pandas_parallel_apply.SeriesParallel`.
    parallelism : {'multithread', 'multiprocess'}
        multi-threading or multi-processing. Passed as-is to
        :class:`pandas_parallel_apply.SeriesParallel`.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes.
    scoped_msg : str, optional
        whether or not to scoped_info the progress bars. Only valid if a logger is provided

    Returns
    -------
    pandas.Series
        output series by invoking `s.apply`.

    See Also
    --------
    pandas_parallel_apply.SeriesParallel
        the wrapped class for the parallel_apply purpose
    """

    if logger:
        sp = SeriesParallel(s, n_cores=n_cores, parallelism=parallelism, pbar=True)
        if scoped_msg:
            context = logger.scoped_info(scoped_msg)
        else:
            context = ctx.nullcontext()
    else:
        sp = SeriesParallel(s, n_cores=n_cores, parallelism=parallelism, pbar=False)
        context = ctx.nullcontext()

    with context:
        return sp.apply(func)


def stats(s: pd.Series) -> dict:
    """Computes some statistics out of a scalar series.

    Parameters
    ----------
    s : pandas.Series
        a scalar series of non-null numeric values

    Returns
    -------
    dict
        an output dictionary containing keys 'min', 'max', 'mean', 'std'
    """

    res = {
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "std": s.std(),
    }
    return res


def int_list(s: pd.Series, dropna: bool = False) -> tp.List[int]:
    """Extracts unique values from the series, converting them to integers.

    Parameters
    ----------
    s : pandas.Series
        a scalar series
    dropna : bool
        whether or not to ignore null values

    Returns
    -------
    list
        list of unique values of the series, converted into integers
    """

    if dropna:
        s = s.dropna()

    return s.astype(int).drop_duplicates().tolist()
