'''Additional utilities dealing with dataframes.'''


import pandas as pd
from tqdm import tqdm


__all__ = ['rename_column', 'row_apply', 'warn_duplicate_records']


def rename_column(df: pd.DataFrame, old_column: str, new_column:str) -> bool:
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


def row_apply(df: pd.DataFrame, func, bar_unit='it') -> pd.DataFrame:
    '''Applies a function on every row of a pandas.DataFrame, optionally with a progress bar.

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
    '''

    if bar_unit is None:
        return df.apply(func, axis=1)

    bar = tqdm(total=len(df), unit=bar_unit)

    def func2(row):
        res = func(row)
        bar.update()
        return res

    with bar:
        return df.apply(func2, axis=1)


def warn_duplicate_records(
        df: pd.DataFrame,
        keys: list,
        msg_format: str = "Detected {dup_cnt}/{rec_cnt} duplicate records.",
        logger=None):
    '''Warns of duplicate records in the dataframe based on a list of keys.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    keys : list
        list of column names
    msg_format : str, optional
        the message to be logged. Two keyword arguments will be provided 'rec_cnt' and 'dup_cnt'.
    logger : logging.Logger, optional
        logger to warn if duplicate records are detected
    '''
    if not logger:
        return

    cnt0 = len(df)
    if not isinstance(keys, list):
        keys = [keys]
    cnt1 = len(df[keys].drop_duplicates())
    if cnt1 < cnt0:
        logger.warning(msg_format.format(dup_cnt=cnt0-cnt1, rec_cnt=cnt0))
