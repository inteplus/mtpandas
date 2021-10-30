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
    {'ndarray', 'SparseNdarray', 'Image', 'object', etc}
        the type of the series. If it is a normal series, the string representing the dtype
        attribute of the series is returned
    '''
    if len(s) == 0:
        return 'object'

    can_be_ndarray = True
    can_be_SparseNdarray = True
    can_be_Image = True
    for x in s.tolist():
        if x is None:
            continue
        if isinstance(x, np.ndarray):
            can_be_SparseNdarray = False
            can_be_Image = False
            if not can_be_ndarray:
                break
            continue
        if isinstance(x, np.SparseNdarray):
            can_be_ndarray = False
            can_be_Image = False
            if not can_be_SparseNdarray:
                break
            continue
        if isinstance(x, cv.Image):
            can_be_ndarray = False
            can_be_SparseNdarray = False
            if not can_be_Image:
                break
            continue
        can_be_ndarray = False
        can_be_Image = False
        break

    if can_be_ndarray:
        return 'ndarray'

    if can_be_SparseNdarray:
        return 'SparseNdarray'

    if can_be_Image:
        return 'Image'

    return str(s.dtype)
