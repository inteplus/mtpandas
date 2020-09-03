'''Additional utilities dealing with dataframes.'''


import pandas as _pd


__all__ = ['rename_column']


def rename_column(df, old_column, new_column):
    '''Renames a column in a dataframe.

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
    '''
    if not old_column in df.columns:
        return False

    columns = list(df.columns)
    columns[columns.index(old_column)] = new_column
    df.columns = columns
    return True
