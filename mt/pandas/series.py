'''Extra functions to augment pandas.Series.'''


import numpy as _np
import pandas as _pd


__all__ = ['Series4json', 'json4Series', 'to_categorical']


def Series4json(obj):
    '''Converts a json-like object in to a pandas.Series.

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
    A json-like object contains 2 array-likes, each has length K. The first array represents the index array. The second array represents the value array.
    '''
    return _pd.Series(index=obj[0], data=obj[1])


def json4Series(obj):
    '''Converts a pandas.Series into a json-like object.

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
    A json-like object contains 2 array-likes, each has length K. The first array represents the index array. The second array represents the value array.
    '''
    return [obj.index.tolist(), obj.tolist()]


def to_categorical(series, value_list, missing_values='raise_exception', logger=None):
    '''Converts a string series representing a categorical field into a zero-indexed categorical field and a one-hot field in json format.

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
    '''
    df = series.to_frame('cat') # to dataframe

    # check for missing values
    df2 = df[~df['cat'].isin(value_list)]
    if len(df2) > 0:
        missing_list = df2['cat'].drop_duplicates().tolist()
        if missing_values == 'raise_exception':
            raise ValueError("Missing values detected: {}".format(missing_list))
        elif missing_values =='remove_and_warn':
            df = df[df['cat'].isin(value_list)]
            if logger:
                logger.warn("Missing values detected and removed: {}".format(missing_list))
        elif missing_values =='remove_in_silence':
            df = df[df['cat'].isin(value_list)]
        else:
            raise ValueError("Unknown policy to deal with missing values: {}. Accepted policies are 'raise_exception', 'remove_and_warn' or 'remove_in_silence'.".format(missing_values))

    # cat id
    df['cat_id'] = df['cat'].map(value_list.index)

    # one hot
    eye = _np.eye(len(value_list))
    eye_list = {i: eye[i] for i in range(len(value_list))}
    df['one_hot'] = df['cat_id'].map(eye_list)

    return df.drop('cat', axis=1)
