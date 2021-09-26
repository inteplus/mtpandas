import io
import json
import pandas as pd
from halo import Halo

from mt import np, cv
from mt.base import path, aio, dummy_scope
from .csv import read_csv_asyn, to_csv_asyn


__all__ = ['dfload_asyn', 'dfload', 'dfsave_asyn', 'dfsave', 'get_dftype', 'dfpack', 'dfunpack']


def array2list(x):
    '''Converts a nested numpy.ndarray object into a nested list object.'''
    return [array2list(y) for y in x] if isinstance(x, np.ndarray) and x.ndim == 1 else x


def get_dftype(s):
    '''Detects the dftype of the series.

    Determine whether a series is an ndarray series or an Image series or a normal series.

    Parameters
    ----------
    s : pandas.Series
        the series to investigate

    Returns
    -------
    {'ndarray', 'Image', 'object'}
        the type of the series
    '''
    if len(s) == 0:
        return 'object'

    can_be_ndarray = True
    can_be_Image = True
    for x in s.tolist():
        if x is None:
            continue
        if isinstance(x, np.ndarray):
            can_be_Image = False
            if not can_be_ndarray:
                break
            continue
        if isinstance(x, cv.Image):
            can_be_ndarray = False
            if not can_be_Image:
                break
            continue
        can_be_ndarray = False
        can_be_Image = False
        break

    if can_be_ndarray:
        return 'ndarray'

    if can_be_Image:
        return 'Image'

    return 'object'


def dfpack(df, spinner=None):
    '''Packs a dataframe into a more compact format.

    At the moment, it converts each ndarray column into 3 columns, and each cv.Image column into a json column.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to be packed
    spinner : Halo, optional
        spinner for tracking purposes

    Returns
    -------
    pandas.DataFrame
        output dataframe
    '''

    df2 = df[[]].copy() # copy the index
    for key in df.columns:
        dftype = get_dftype(df[key])

        if dftype == 'ndarray':
            if spinner is not None:
                spinner.text = "packing ndarray field '{}'".format(key)
            df2[key+'_df_nd_ravel'] = df[key].apply(lambda x: None if x is None else x.ravel())
            df2[key+'_df_nd_shape'] = df[key].apply(lambda x: None if x is None else np.array(x.shape))
            df2[key+'_df_nd_dtype'] = df[key].apply(lambda x: None if x is None else x.dtype.str)

        elif dftype == 'Image':
            if spinner is not None:
                spinner.text = "packing Image field '{}'".format(key)
            df2[key+'_df_imm'] = df[key].apply(lambda x: None if x is None else json.dumps(x.to_json()))

        else:
            if spinner is not None:
                spinner.text = "passing field '{}'".format(key)
            df2[key] = df[key]

    return df2


def dfunpack(df, spinner=None):
    '''Unpacks a compact dataframe into a more expanded format.

    This is the reverse function of :func:`dfpack`.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to be unpacked
    spinner : Halo, optional
        spinner for tracking purposes

    Returns
    -------
    pandas.DataFrame
        output dataframe
    '''

    key2 = '' # just to trick pylint
    df2 = df[[]].copy() # copy the index
    for key in df.columns:
        if key.endswith('_df_imm'):
            key2 = key[:-7]
            if spinner is not None:
                spinner.text = "unpacking Image field '{}'".format(key2)
            df2[key2] = df[key].apply(lambda x: None if x is None else cv.Image.from_json(json.loads(x)))
        elif key.endswith('_df_nd_ravel'):
            key2 = key[:-12]
            if spinner is not None:
                spinner.text = "unpacking ndarray field '{}'".format(key2)
            def unpack_ndarray(row):
                ravel = row[key2+'_df_nd_ravel']
                dtype = np.dtype(row[key2+'_df_nd_dtype'])
                if isinstance(ravel, np.ndarray): # already a 1D array?
                    ravel = ravel.astype(dtype)
                else: # list or something else?
                    ravel = np.array(ravel, dtype=dtype)
                return ravel.reshape(row[key2+'_df_nd_shape'])
            df2[key2] = df.apply(unpack_ndarray, axis=1)
        elif '_df_nd_' in key:
            continue
        else:
            if spinner is not None:
                spinner.text = "passing field '{}'".format(key)
            df2[key] = df[key]

    return df2


def determine_parquet_engine():
    try: # unlike pandas, we prioritise fastparquet over pyarrow
        import fastparquet
        return 'fastparquet'
    except ImportError:
        return 'auto'

async def dfload_asyn(df_filepath, *args, show_progress=False, unpack=True, parquet_convert_ndarray_to_list=False, context_vars: dict = {}, **kwargs):
    '''An asyn function that loads a dataframe file based on the file's extension.

    Parameters
    ----------
    df_filepath : str
        local path to an existing dataframe. The file extension is used to determine the file type.
    show_progress : bool
        show a progress spinner in the terminal
    unpack : bool
        whether or not to unpack the dataframe after loading
    parquet_convert_ndarray_to_list : bool
        whether or not to convert 1D ndarrays in the loaded parquet table into Python lists
    args : list
        list of positional arguments to pass to the corresponding reader
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
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

    filepath = df_filepath.lower()

    if filepath.endswith('.parquet'):
        if show_progress:
            spinner = Halo("dfloading '{}'".format(filepath), spinner='dots')
            scope = spinner
        else:
            spinner = None
            scope = dummy_scope
        with scope:
            try:
                data = await aio.read_binary(df_filepath, context_vars=context_vars)
                kwargs = kwargs.copy()
                if 'engine' not in kwargs:
                    kwargs['engine'] = determine_parquet_engine()
                df = pd.read_parquet(io.BytesIO(data), *args, **kwargs)

                if parquet_convert_ndarray_to_list:
                    for x in df.columns:
                        if show_progress:
                            spinner.text = 'converting column: {}'.format(x)
                        if df.dtypes[x] == np.dtype('O'): # object
                            df[x] = df[x].apply(array2list) # because Parquet would save lists into nested numpy arrays which is not we expect yet.

                if unpack:
                    df = dfunpack(df, spinner=spinner)

                if show_progress:
                    spinner.succeed("dfloaded '{}'".format(filepath))
            except:
                if show_progress:
                    spinner.fail("failed to dfload '{}'".format(filepath))
                raise

        return df


    if filepath.endswith('.csv') or filepath.endswith('.csv.zip'):
        df = await read_csv_asyn(df_filepath, *args, show_progress=show_progress, context_vars=context_vars, **kwargs)

        if unpack:
            df = dfunpack(df)

        return df

    raise TypeError("Unknown file type: '{}'".format(df_filepath))


def dfload(df_filepath, *args, show_progress=False, unpack=True, parquet_convert_ndarray_to_list=False, **kwargs):
    '''Loads a dataframe file based on the file's extension.

    Parameters
    ----------
    df_filepath : str
        local path to an existing dataframe. The file extension is used to determine the file type.
    show_progress : bool
        show a progress spinner in the terminal
    unpack : bool
        whether or not to unpack the dataframe after loading
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
    return aio.srun(dfload_asyn, df_filepath, *args, show_progress=show_progress, unpack=unpack, parquet_convert_ndarray_to_list=parquet_convert_ndarray_to_list, **kwargs)


async def dfsave_asyn(df, df_filepath, file_mode=0o664, show_progress=False, pack=True, context_vars: dict = {}, file_write_delayed: bool = False, **kwargs):
    '''An asyn function that saves a dataframe to a file based on the file's extension.

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
    pack : bool
        whether or not to pack the dataframe before saving
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    file_write_delayed : bool
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual.
    kwargs : dict
        dictionary of keyword arguments to pass to the corresponding writer

    Returns
    -------
    asyncio.Future or int
        either a future or the number of bytes written, depending on whether the file write
        task is delayed or not

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.to_csv`. For '.parquet' files, we use :func:`pandas.DataFrame.to_parquet`.

    Raises
    ------
    TypeError
        if file type is unknown or if the input is not a dataframe
    '''

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas.DataFrame. Got '{}'.".format(type(df)))

    filepath = df_filepath.lower()

    if filepath.endswith('.parquet'):
        if show_progress:
            spinner = Halo(text="dfsaving '{}'".format(filepath), spinner='dots')
            scope = spinner
        else:
            spinner = None
            scope = dummy_scope
        with scope:
            try:
                if pack:
                    df = dfpack(df, spinner=spinner)

                kwargs = kwargs.copy()
                if 'engine' not in kwargs:
                    kwargs['engine'] = determine_parquet_engine()
                if kwargs['engine'] != 'fastparquet' and not 'use_deprecated_int96_timestamps' in kwargs:
                    kwargs['use_deprecated_int96_timestamps'] = True # to avoid exception pyarrow.lib.ArrowInvalid: Casting from timestamp[ns] to timestamp[ms] would lose data: XXXXXXX
                data = df.to_parquet(None, **kwargs)
                res = await aio.write_binary(df_filepath, data, file_mode=file_mode, context_vars=context_vars, file_write_delayed=file_write_delayed)

                if show_progress:
                    spinner.succeed("dfsaved '{}'".format(filepath))
            except:
                if show_progress:
                    spinner.fail("failed to dfsave '{}'".format(filepath))
                raise
        return res

    if filepath.endswith('.csv') or filepath.endswith('.csv.zip'):
        if pack:
            df = dfpack(df)
        res = await to_csv_asyn(df, df_filepath, file_mode=file_mode, show_progress=show_progress, context_vars=context_vars, **kwargs)
        return res

    raise TypeError("Unknown file type: '{}'".format(df_filepath))


def dfsave(df, df_filepath, file_mode=0o664, show_progress=False, pack=True, **kwargs):
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
    pack : bool
        whether or not to pack the dataframe before saving
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
    return aio.srun(dfsave_asyn, df, df_filepath, file_mode=file_mode, show_progress=show_progress, pack=pack, **kwargs)
