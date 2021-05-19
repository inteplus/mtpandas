import numpy as _np
import pandas as _pd
from halo import Halo
from mt.base.with_utils import dummy_scope
import mt.base.path as _p
from .csv import read_csv, to_csv


__all__ = ['dfload', 'dfsave']


def array2list(x):
    '''Converts a nested numpy.ndarray object into a nested list object.'''
    return [array2list(y) for y in x] if isinstance(x, _np.ndarray) and x.ndim == 1 else x


def dfload(df_filepath, *args, show_progress=False, parquet_convert_ndarray_to_list=False, **kwargs):
    '''Loads a dataframe file based on the file's extension.

    Parameters
    ----------
    df_filepath : str
        local path to an existing dataframe. The file extension is used to determine the file type.
    show_progress : bool
        show a progress spinner in the terminal
    parquet_convert_ndarray_to_list : bool
        whether or not to convert 1D ndarrays in the loaded parquet table into Python lists
    args : list
        list of positional arguments to pass to the corresponding reader
    kwargs : dict
        dictionary of keyword arguments to pass to the corresponding reader

    Returns
    -------
    pandas.DataFrame
        loaded dataframe

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.read_csv`. For '.parquet' files, we use :func:`pandas.read_parquet`.

    Raises
    ------
    TypeError
        if file type is unknown
    '''

    path = df_filepath.lower()

    if path.endswith('.parquet'):
        spinner = Halo("dfloading '{}'".format(path), spinner='dots') if show_progress else dummy_scope
        with spinner:
            try:
                df = _pd.read_parquet(df_filepath, *args, **kwargs)

                if parquet_convert_ndarray_to_list:
                    for x in df.columns:
                        if isinstance(spinner, Halo):
                            spinner.text = 'converting column: {}'.format(x)
                        if df.dtypes[x] == _np.dtype('O'): # object
                            df[x] = df[x].apply(array2list) # because Parquet would save lists into nested numpy arrays which is not we expect yet.

                spinner.succeed("dfloaded '{}'".format(path))
            except:
                if isinstance(spinner, Halo):
                    spinner.fail("failed to dfload '{}'".format(path))
                raise

        return df


    if path.endswith('.csv') or path.endswith('.csv.zip'):
        return read_csv(df_filepath, *args, show_progress=show_progress, **kwargs)

    raise TypeError("Unknown file type: '{}'".format(df_filepath))


def dfsave(df, df_filepath, file_mode=0o664, show_progress=False, **kwargs):
    '''Saves a dataframe to a file based on the file's extension.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    df_filepath : str
        local path to an existing dataframe. The file extension is used to determine the file type.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode will happen.
    show_progress : bool
        show a progress spinner in the terminal
    kwargs : dict
        dictionary of keyword arguments to pass to the corresponding writer

    Returns
    -------
    object
        whatever the corresponding writer returns

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.to_csv`. For '.parquet' files, we use :func:`pandas.DataFrame.to_parquet`.

    Raises
    ------
    TypeError
        if file type is unknown or if the input is not a dataframe
    '''

    if not isinstance(df, _pd.DataFrame):
        raise TypeError("Input must be a pandas.DataFrame. Got '{}'.".format(type(df)))

    path = df_filepath.lower()

    if path.endswith('.parquet'):
        spinner = Halo(text="dfsaving '{}'".format(path), spinner='dots') if show_progress else dummy_scope
        with spinner:
            try:
                if not 'use_deprecated_int96_timestamps' in kwargs:
                    kwargs = kwargs.copy()
                    kwargs['use_deprecated_int96_timestamps'] = True # to avoid exception pyarrow.lib.ArrowInvalid: Casting from timestamp[ns] to timestamp[ms] would lose data: XXXXXXX
                res = df.to_parquet(df_filepath, **kwargs)
                if file_mode:  # chmod
                    _p.chmod(df_filepath, file_mode)

                if isinstance(spinner, Halo):
                    spinner.succeed("dfsaved '{}'".format(path))
            except:
                if isinstance(spinner, Halo):
                    spinner.fail("failed to dfsave '{}'".format(path))
                raise
        return res

    if path.endswith('.csv') or path.endswith('.csv.zip'):
        return to_csv(df, df_filepath, file_mode=file_mode, show_progress=show_progress, **kwargs)

    raise TypeError("Unknown file type: '{}'".format(df_filepath))
