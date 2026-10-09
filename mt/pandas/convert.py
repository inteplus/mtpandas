"""Loading and saving dataframes to files whose format is chosen by the file extension.

The main functions are :func:`dfload` and :func:`dfsave` (and their asyn versions) which support
'.csv', '.csv.zip', '.parquet' and '.pdh5' files. Columns that are not natively supported by the
file formats (ndarray and Image columns) are converted by :func:`dfpack` before saving and
restored by :func:`dfunpack` after loading.
"""

import warnings
import io
import json
import pandas as pd

from mt import tp, np, cv, ctx, aio
from mt.halo import HaloAuto

from .csv import read_csv_asyn, to_csv_asyn
from .dftype import get_dftype
from .pdh5 import load_pdh5_asyn, save_pdh5, Pdh5Cell


__all__ = [
    "dfload_asyn",
    "dfload",
    "dfsave_asyn",
    "dfsave",
    "dfpack",
    "dfunpack",
    "Pdh5Cell",
]


def array2list(x):
    """Converts a nested numpy.ndarray object into a nested list object.

    Only 1D ndarrays (of ndarrays) are converted recursively. Any other object is returned as is.
    It is meant to fix nested arrays that parquet produces when reading list columns.
    """
    return (
        [array2list(y) for y in x] if isinstance(x, np.ndarray) and x.ndim == 1 else x
    )


def dfpack(df, spinner=None):
    """Packs a dataframe into a more compact format.

    At the moment, it converts each ndarray column `key` into 3 columns `key_df_nd_ravel` (the
    flattened array), `key_df_nd_shape` and `key_df_nd_dtype`, and each cv.Image column `key` into
    a json-string column `key_df_imm`. Other columns are passed through. Column types are detected
    using :func:`mt.pandas.dftype.get_dftype`.

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

    See Also
    --------
    dfunpack
        the reverse function

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mt.pandas import dfpack
    >>> df = pd.DataFrame({"a": [1, 2], "v": [np.zeros((2, 2)), np.ones((1, 2))]})
    >>> dfpack(df).columns.tolist()
    ['a', 'v_df_nd_ravel', 'v_df_nd_shape', 'v_df_nd_dtype']
    """

    with warnings.catch_warnings(record=True) as l_msgs:
        df2 = df[[]].copy()  # copy the index
        for key in df.columns:
            dftype = get_dftype(df[key])

            if dftype == "ndarray":
                if spinner is not None:
                    spinner.text = f"packing ndarray field '{key}'"
                df2[key + "_df_nd_ravel"] = df[key].apply(
                    lambda x: None if x is None else x.ravel()
                )
                df2[key + "_df_nd_shape"] = df[key].apply(
                    lambda x: None if x is None else np.array(x.shape)
                )
                df2[key + "_df_nd_dtype"] = df[key].apply(
                    lambda x: None if x is None else x.dtype.str
                )

            elif dftype == "Image":
                if spinner is not None:
                    spinner.text = f"packing Image field '{key}'"
                df2[key + "_df_imm"] = df[key].apply(
                    lambda x: None if x is None else json.dumps(x.to_json())
                )

            else:
                if spinner is not None:
                    spinner.text = f"passing field '{key}'"
                df2[key] = df[key]

    if l_msgs:
        to_copy = False
        for msg in l_msgs:
            if issubclass(msg.category, pd.errors.PerformanceWarning):
                to_copy = True
            else:
                warnings.warn(msg)

        if to_copy:
            if spinner is not None:
                spinner.text = "defragmenting the dataframe"
            df2 = df2.copy()

    return df2


def dfunpack(df, spinner=None):
    """Unpacks a compact dataframe into a more expanded format.

    This is the reverse function of :func:`dfpack`. Columns ending with ``_df_nd_ravel``,
    ``_df_nd_shape`` and ``_df_nd_dtype`` are merged back into an ndarray column, columns ending
    with ``_df_imm`` are converted back into cv.Image columns, and other columns are passed
    through. Null items stay None.

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

    See Also
    --------
    dfpack
        the reverse function

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mt.pandas import dfpack, dfunpack
    >>> df = pd.DataFrame({"a": [1, 2], "v": [np.zeros((2, 2)), np.ones((1, 2))]})
    >>> df2 = dfunpack(dfpack(df))
    >>> df2.columns.tolist()
    ['a', 'v']
    >>> [x.shape for x in df2["v"]]
    [(2, 2), (1, 2)]
    """

    key2 = ""  # just to trick pylint
    df2 = df[[]].copy()  # copy the index
    for key in df.columns:
        if key.endswith("_df_imm"):
            key2 = key[:-7]
            if spinner is not None:
                spinner.text = f"unpacking Image field '{key2}'"
            df2[key2] = df[key].apply(
                lambda x: None if x is None else cv.Image.from_json(json.loads(x))
            )
        elif key.endswith("_df_nd_ravel"):
            key2 = key[:-12]
            if spinner is not None:
                spinner.text = f"unpacking ndarray field '{key2}'"

            def unpack_ndarray(row):
                ravel = row[key2 + "_df_nd_ravel"]
                if ravel is None or (
                    not isinstance(ravel, (np.ndarray, list)) and pd.isna(ravel)
                ):
                    return None
                dtype_val = row[key2 + "_df_nd_dtype"]
                if dtype_val is None or (
                    not isinstance(dtype_val, str) and pd.isna(dtype_val)
                ):
                    return None
                dtype = np.dtype(dtype_val)
                if isinstance(ravel, np.ndarray):  # already a 1D array?
                    ravel = ravel.astype(dtype)
                else:  # list or something else?
                    ravel = np.array(ravel, dtype=dtype)
                return ravel.reshape(row[key2 + "_df_nd_shape"])

            df2[key2] = df.apply(unpack_ndarray, axis=1)
        elif "_df_nd_" in key:
            continue
        else:
            if spinner is not None:
                spinner.text = f"passing field '{key}'"
            df2[key] = df[key]

    return df2


async def dfload_asyn(
    df_filepath,
    *args,
    show_progress=False,
    unpack=True,
    parquet_convert_ndarray_to_list=False,
    file_read_delayed: bool = False,
    max_rows: tp.Optional[int] = None,
    nrows: tp.Optional[int] = None,
    context_vars: dict = {},
    **kwargs
) -> pd.DataFrame:
    """An asyn function that loads a dataframe file based on the file's extension.

    Supported extensions (case-insensitive) are '.csv', '.csv.zip', '.parquet' and '.pdh5'.

    Parameters
    ----------
    df_filepath : str
        local path to an existing dataframe. The file extension is used to determine the file type.
    show_progress : bool
        show a progress spinner in the terminal. Default is False.
    unpack : bool
        whether or not to unpack the dataframe after loading (see :func:`dfunpack`). Default is
        True. Ignored for '.pdh5' format.
    parquet_convert_ndarray_to_list : bool
        whether or not to convert 1D ndarrays in the loaded parquet table into Python lists.
        Ignored for '.pdh5' format.
    file_read_delayed : bool
        whether or not some columns can be delayed for reading later. Only valid for '.pdh5'
        format.
    max_rows : int, optional
        limit the maximum number of rows to read. Only valid for '.csv', '.pdh5' and '.parquet'
        formats. This argument is only for backward compatibility. Please use nrows instead.
    nrows : int, optional
        limit the maximum number of rows to read. Only valid for '.csv', '.pdh5' and '.parquet'
        formats.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
        Ignored for '.pdh5' format.
    *args : tuple
        list of positional arguments to pass to the corresponding reader. Ignored for '.pdh5'
        format.
    **kwargs : dict
        dictionary of keyword arguments to pass to the corresponding reader. Ignored for '.pdh5'
        format.

    Returns
    -------
    pandas.DataFrame
        loaded dataframe

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.read_csv`. For '.parquet' files, we
    use :func:`pandas.read_parquet`. For `.pdh5` files, we use
    :func:`mt.pandas.pdh5.load_pdh5_asyn`.

    When `nrows` is given for a '.parquet' file, only the first batch of `nrows` rows is read and
    the positional and keyword arguments for the reader are ignored.

    Raises
    ------
    TypeError
        if file type is unknown

    Examples
    --------
    >>> import asyncio
    >>> import os
    >>> import tempfile
    >>> import pandas as pd
    >>> from mt.pandas import dfsave, dfload_asyn
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    >>> with tempfile.TemporaryDirectory() as d:
    ...     filepath = os.path.join(d, "df.parquet")
    ...     _ = dfsave(df, filepath)
    ...     df2 = asyncio.run(dfload_asyn(filepath, context_vars={"async": True}))
    >>> df2
       a  b
    0  1  x
    1  2  y
    2  3  z
    """

    if max_rows is not None:
        msg = (
            "In function dfload_asyn, argument 'max_rows' is deprecated and should be "
            "replaced by 'nrows'."
        )
        warnings.warn(msg)
        nrows = max_rows

    filepath = df_filepath.lower()

    if filepath.endswith(".pdh5"):
        return await load_pdh5_asyn(
            df_filepath,
            show_progress=show_progress,
            file_read_delayed=file_read_delayed,
            max_rows=nrows,
            context_vars=context_vars,
            **kwargs
        )

    if filepath.endswith(".parquet"):
        if show_progress:
            spinner = HaloAuto(f"dfloading '{filepath}'", spinner="dots")
            scope = spinner
        else:
            spinner = None
            scope = ctx.nullcontext()
        with scope:
            try:
                if nrows is None:
                    data = await aio.read_binary(df_filepath, context_vars=context_vars)
                    df = pd.read_parquet(io.BytesIO(data), *args, **kwargs)
                else:
                    try:
                        from pyarrow.parquet import ParquetFile
                        import pyarrow as pa

                        pf = ParquetFile(df_filepath)
                        rows = next(pf.iter_batches(batch_size=nrows))
                        df = pa.Table.from_batches([rows]).to_pandas()
                    except ImportError:
                        if show_progress:
                            spinner.text = (
                                "PyArrow is not available. Loading the whole file."
                            )
                        data = await aio.read_binary(
                            df_filepath, context_vars=context_vars
                        )
                        df = pd.read_parquet(io.BytesIO(data), *args, **kwargs)

                if parquet_convert_ndarray_to_list:
                    for x in df.columns:
                        if show_progress:
                            spinner.text = f"converting column: {x}"
                        if df.dtypes[x] == np.dtype("O"):  # object
                            # because Parquet would save lists into nested numpy arrays
                            # which is not we expect yet.
                            df[x] = df[x].apply(array2list)

                if unpack:
                    df = dfunpack(df, spinner=spinner)

                if show_progress:
                    spinner.succeed(f"dfloaded '{filepath}'")
            except:
                if show_progress:
                    spinner.fail(f"failed to dfload '{filepath}'")
                raise

        return df

    if filepath.endswith(".csv") or filepath.endswith(".csv.zip"):
        df = await read_csv_asyn(
            df_filepath,
            *args,
            nrows=nrows,
            show_progress=show_progress,
            context_vars=context_vars,
            **kwargs
        )

        if unpack:
            df = dfunpack(df)

        return df

    raise TypeError(f"Unknown file type: '{df_filepath}'")


def dfload(
    df_filepath,
    *args,
    show_progress=False,
    unpack=True,
    parquet_convert_ndarray_to_list=False,
    file_read_delayed: bool = False,
    max_rows: tp.Optional[int] = None,
    nrows: tp.Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """Loads a dataframe file based on the file's extension.

    Supported extensions (case-insensitive) are '.csv', '.csv.zip', '.parquet' and '.pdh5'. This is
    the synchronous version of :func:`dfload_asyn`.

    Parameters
    ----------
    df_filepath : str
        local path to an existing dataframe. The file extension is used to determine the file type.
    show_progress : bool
        show a progress spinner in the terminal. Default is False.
    unpack : bool
        whether or not to unpack the dataframe after loading (see :func:`dfunpack`). Default is
        True. Ignored for '.pdh5' format.
    parquet_convert_ndarray_to_list : bool
        whether or not to convert 1D ndarrays in the loaded parquet table into Python lists.
        Ignored for '.pdh5' format.
    file_read_delayed : bool
        whether or not some columns can be delayed for reading later. Only valid for '.pdh5'
        format.
    max_rows : int, optional
        limit the maximum number of rows to read. Only valid for '.csv', '.pdh5' and '.parquet'
        formats. This argument is only for backward compatibility. Please use nrows instead.
    nrows : int, optional
        limit the maximum number of rows to read. Only valid for '.csv', '.pdh5' and '.parquet'
        formats.
    *args : tuple
        list of positional arguments to pass to the corresponding reader. Ignored for '.pdh5'
        format.
    **kwargs : dict
        dictionary of keyword arguments to pass to the corresponding reader. Ignored for '.pdh5'
        format.

    Returns
    -------
    pandas.DataFrame
        loaded dataframe

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.read_csv`. For '.parquet' files, we
    use :func:`pandas.read_parquet`. For `.pdh5` files, we use
    :func:`mt.pandas.pdh5.load_pdh5_asyn`.

    When `nrows` is given for a '.parquet' file, only the first batch of `nrows` rows is read and
    the positional and keyword arguments for the reader are ignored. A '.csv' file saved by
    :func:`dfsave` has its column dtypes restored from the '.meta' file next to it.

    Raises
    ------
    TypeError
        if file type is unknown

    See Also
    --------
    dfsave
        the reverse function

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import pandas as pd
    >>> from mt.pandas import dfsave, dfload
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    >>> with tempfile.TemporaryDirectory() as d:
    ...     filepath = os.path.join(d, "df.parquet")
    ...     _ = dfsave(df, filepath)
    ...     df2 = dfload(filepath)
    ...     df3 = dfload(filepath, nrows=2)
    >>> df2
       a  b
    0  1  x
    1  2  y
    2  3  z
    >>> len(df3)
    2
    """
    return aio.srun(
        dfload_asyn,
        df_filepath,
        *args,
        show_progress=show_progress,
        unpack=unpack,
        parquet_convert_ndarray_to_list=parquet_convert_ndarray_to_list,
        file_read_delayed=file_read_delayed,
        max_rows=max_rows,
        nrows=nrows,
        **kwargs
    )


async def dfsave_asyn(
    df,
    df_filepath,
    file_mode=0o664,
    show_progress=False,
    pack=True,
    context_vars: dict = {},
    file_write_delayed: bool = False,
    make_dirs: bool = False,
    **kwargs
):
    """An asyn function that saves a dataframe to a file based on the file's extension.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    df_filepath : str
        local path to the file to write. The file extension is used to determine the file type.
        Supported extensions (case-insensitive) are '.csv', '.csv.zip', '.parquet' and '.pdh5'.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen. Default is 0o664.
    show_progress : bool
        show a progress spinner in the terminal. Default is False.
    pack : bool
        whether or not to pack the dataframe before saving (see :func:`dfpack`). Default is True.
        Ignored for '.pdh5' format.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
        Ignored for '.pdh5' format.
    file_write_delayed : bool
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual. Ignored for '.pdh5' format.
    make_dirs : bool
        Whether or not to make the folders containing the path before writing to the file.
    **kwargs : dict
        dictionary of keyword arguments to pass to the corresponding writer. Ignored for '.pdh5'
        format.

    Returns
    -------
    asyncio.Future or int
        either a future or the number of bytes written, depending on whether the file write
        task is delayed or not. For '.pdh5' format, 1 is returned.

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.to_csv`. For '.parquet' files, we
    use :func:`pandas.DataFrame.to_parquet`. For '.pdh5' files, we use
    :func:`mt.pandas.pdh5.save_pdh5`.

    When you want to save a large parquet file, you may want to pass `row_group_size` as a keyword
    argument for pyarrow or `row_group_offsets` as a keyword argument for fastparquet. This would
    allow the file to be read in smaller chunks later on, which can save memory and speed up the
    reading process. The optimal value for `row_group_size` or `row_group_offsets` depends on your
    specific use case and system configuration, but a common choice is around 10000 to 50000.

    Raises
    ------
    TypeError
        if file type is unknown or if the input is not a dataframe

    Examples
    --------
    >>> import asyncio
    >>> import os
    >>> import tempfile
    >>> import pandas as pd
    >>> from mt.pandas import dfsave_asyn, dfload
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> with tempfile.TemporaryDirectory() as d:
    ...     filepath = os.path.join(d, "df.parquet")
    ...     n_bytes = asyncio.run(dfsave_asyn(df, filepath, context_vars={"async": True}))
    ...     df2 = dfload(filepath)
    >>> n_bytes > 0
    True
    >>> df2["a"].tolist()
    [1, 2, 3]
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas.DataFrame. Got '{type(df)}'.")

    filepath = df_filepath.lower()

    if filepath.endswith(".pdh5"):
        save_pdh5(
            df_filepath, df, file_mode=file_mode, show_progress=show_progress, **kwargs
        )
        return 1

    if filepath.endswith(".parquet"):
        if show_progress:
            spinner = HaloAuto(text=f"dfsaving '{filepath}'", spinner="dots")
            scope = spinner
        else:
            spinner = None
            scope = ctx.nullcontext()
        with scope:
            try:
                if pack:
                    df = dfpack(df, spinner=spinner)

                kwargs = kwargs.copy()
                if not "use_deprecated_int96_timestamps" in kwargs:
                    # to avoid exception pyarrow.lib.ArrowInvalid: Casting from timestamp[ns]
                    # to timestamp[ms] would lose data: XXXXXXX
                    kwargs["use_deprecated_int96_timestamps"] = True
                data = df.to_parquet(None, **kwargs)
                res = await aio.write_binary(
                    df_filepath,
                    data,
                    file_mode=file_mode,
                    context_vars=context_vars,
                    file_write_delayed=file_write_delayed,
                    make_dirs=make_dirs,
                )

                if show_progress:
                    spinner.succeed(f"dfsaved '{filepath}'")
            except:
                if show_progress:
                    spinner.fail(f"failed to dfsave '{filepath}'")
                raise
        return res

    if filepath.endswith(".csv") or filepath.endswith(".csv.zip"):
        if pack:
            df = dfpack(df)
        res = await to_csv_asyn(
            df,
            df_filepath,
            file_mode=file_mode,
            show_progress=show_progress,
            context_vars=context_vars,
            **kwargs
        )
        return res

    raise TypeError(f"Unknown file type: '{df_filepath}'")


def dfsave(
    df,
    df_filepath,
    file_mode=0o664,
    show_progress=False,
    pack=True,
    make_dirs: bool = False,
    **kwargs
):
    """Saves a dataframe to a file based on the file's extension.

    Supported extensions (case-insensitive) are '.csv', '.csv.zip', '.parquet' and '.pdh5'. This is
    the synchronous version of :func:`dfsave_asyn`.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe
    df_filepath : str
        local path to the file to write. The file extension is used to determine the file type.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen. Default is 0o664.
    show_progress : bool
        show a progress spinner in the terminal. Default is False.
    pack : bool
        whether or not to pack the dataframe before saving (see :func:`dfpack`). Default is True.
        Ignored for '.pdh5' format.
    make_dirs : bool
        Whether or not to make the folders containing the path before writing to the file.
        Default is False.
    **kwargs : dict
        dictionary of keyword arguments to pass to the corresponding writer

    Returns
    -------
    object
        whatever the corresponding writer returns, typically the number of bytes written. For the
        '.pdh5' format, 1 is returned.

    Notes
    -----
    For '.csv' or '.csv.zip' files, we use :func:`mt.pandas.csv.to_csv`. For '.parquet' files, we
    use :func:`pandas.DataFrame.to_parquet`.

    When you want to save a large parquet file, you may want to pass `row_group_size` as a keyword
    argument for pyarrow or `row_group_offsets` as a keyword argument for fastparquet. This would
    allow the file to be read in smaller chunks later on, which can save memory and speed up the
    reading process. The optimal value for `row_group_size` or `row_group_offsets` depends on your
    specific use case and system configuration, but a common choice is around 10000 to 50000.

    Raises
    ------
    TypeError
        if file type is unknown or if the input is not a dataframe

    See Also
    --------
    dfload
        the reverse function

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import pandas as pd
    >>> from mt.pandas import dfsave, dfload
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> with tempfile.TemporaryDirectory() as d:
    ...     filepath = os.path.join(d, "df.parquet")
    ...     _ = dfsave(df, filepath)
    ...     df2 = dfload(filepath)
    >>> df2["a"].tolist()
    [1, 2, 3]
    """
    return aio.srun(
        dfsave_asyn,
        df,
        df_filepath,
        file_mode=file_mode,
        show_progress=show_progress,
        pack=pack,
        make_dirs=make_dirs,
        **kwargs
    )
