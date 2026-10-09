"""Detection of the logical column type (dftype) of a pandas series."""

import pandas as pd

from mt import np, cv


__all__ = ['isnull', 'get_dftype']


def isnull(obj):
    """Checks whether a scalar is None, pandas.NaT or a float NaN.

    Parameters
    ----------
    obj : object
        a scalar to check

    Returns
    -------
    bool
        whether or not `obj` is null

    Examples
    --------
    >>> from mt.pandas import isnull
    >>> [isnull(x) for x in (None, float("nan"), 0, "")]
    [True, True, False, False]
    """
    if obj is None or obj is pd.NaT:
        return True
    if isinstance(obj, float) and np.isnan(obj):
        return True
    return False


def get_dftype(s):
    '''Detects the dftype of the series.

    Determine whether a series is a json series (lists or dicts), an ndarray series, a sparse
    ndarray series, an Image series, a string series, a Timestamp series, a Timedelta series or a
    normal series. Null items are ignored. The series is classified by looking at its items, and a
    series of mixed kinds is classified as 'object' (or its dtype string).

    Parameters
    ----------
    s : pandas.Series
        the series to investigate

    Returns
    -------
    {'json', 'ndarray', 'SparseNdarray', 'Image', 'str', 'Timestamp','Timedelta', 'object',
    'none', etc}
        the type of the series. An empty series is 'object', a series whose items are all null is
        'none'. If it is a normal series, the string representing the dtype attribute of the
        series is returned, e.g. 'int64'. A series of object dtype made of numbers only is
        'float64'.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mt.pandas import get_dftype
    >>> get_dftype(pd.Series(["a", None]))
    'str'
    >>> get_dftype(pd.Series([[1], {"a": 2}]))
    'json'
    >>> get_dftype(pd.Series([np.zeros(2)]))
    'ndarray'
    >>> get_dftype(pd.Series([1, 2]))
    'int64'
    >>> get_dftype(pd.Series([None, None]))
    'none'
    >>> get_dftype(pd.Series([1, "a"]))
    'object'
    '''
    if len(s) == 0:
        return 'object'

    dftype = None
    for x in s.tolist():
        if isnull(x):
            continue
        if isinstance(x, str):
            if dftype is None:
                dftype = 'str'
            elif dftype != 'str':
                break
            continue
        if isinstance(x, (list, dict)):
            if dftype is None:
                dftype = 'json'
            elif dftype != 'json':
                break
            continue
        if isinstance(x, np.ndarray):
            if dftype is None:
                dftype = 'ndarray'
            elif dftype != 'ndarray':
                break
            continue
        if isinstance(x, np.SparseNdarray):
            if dftype is None:
                dftype = 'SparseNdarray'
            elif dftype != 'SparseNdarray':
                break
            continue
        if isinstance(x, cv.Image):
            if dftype is None:
                dftype = 'Image'
            elif dftype != 'Image':
                break
            continue
        if isinstance(x, pd.Timestamp):
            if dftype is None:
                dftype = 'Timestamp'
            elif dftype != 'Timestamp':
                break
            continue
        if isinstance(x, pd.Timedelta):
            if dftype is None:
                dftype = 'Timedelta'
            elif dftype != 'Timedelta':
                break
            continue
        dftype = 'object'
        break

    if dftype is None:
        return 'none'

    if dftype != 'object':
        return dftype

    dftype = str(s.dtype)
    if dftype != 'object':
        return dftype

    # one last attempt
    types = s.apply(type).unique()
    is_numeric = True
    for x in types:
        if not pd.api.types.is_numeric_dtype(x):
            is_numeric = False
            break
    return 'float64' if is_numeric else 'object'
