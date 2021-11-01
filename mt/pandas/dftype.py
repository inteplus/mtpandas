import pandas as pd

from mt import np, cv


__all__ = ['get_dftype']


def get_dftype(s):
    '''Detects the dftype of the series.

    Determine whether a series is an ndarray series, a sparse ndarray series, an Image series or a
    normal series.

    Parameters
    ----------
    s : pandas.Series
        the series to investigate

    Returns
    -------
    {'json', 'ndarray', 'SparseNdarray', 'Image', 'str', 'Timestamp','Timedelta', 'object', 'none', etc}
        the type of the series. If it is a normal series, the string representing the dtype
        attribute of the series is returned
    '''
    if len(s) == 0:
        return 'object'

    dftype = None
    for x in s.tolist():
        if x is None or x is pd.NaT:
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

    return str(s.dtype)
