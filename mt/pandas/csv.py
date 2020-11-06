import dask.dataframe as _dd
import pandas as _pd
import numpy as _np
from tqdm import tqdm
import csv as _csv
import mt.base.path as _p
import json as _js
import time as _t
import zipfile as _zf


__all__ = ['metadata', 'metadata2dtypes', 'read_csv', 'to_csv']


def metadata(df):
    '''Extracts the metadata of a dataframe.

    Parameters
    ----------
        df : pandas.DataFrame

    Returns
    -------
        meta : json-like
            metadata describing the dataframe
    '''
    meta = {}
    if list(df.index.names) != [None]:  # has index
        index_names = list(df.index.names)
        df = df.reset_index(drop=False)
    else:  # no index
        index_names = []

    meta = {}
    for x in df.dtypes.index:
        dtype = df.dtypes.loc[x]
        name = dtype.name
        if name != 'category':
            meta[x] = name
        else:
            meta[x] = ['category', df[x].cat.categories.tolist(),
                       df[x].cat.ordered]
    meta = {'columns': meta, 'index_names': index_names}
    return meta


def metadata2dtypes(meta):
    '''Creates a dictionary of dtypes from the metadata returned by metadata() function.'''
    res = {}
    s = meta['columns']
    for x in s:
        y = s[x]
        if y == 'datetime64[ns]':
            y = 'object'
        elif isinstance(y, list) and y[0] == 'category':
            y = 'object'
        res[x] = np.dtype(y)
    return res
    # return {x:np.dtype(y) for (x,y) in s.items()}


def read_csv(path, show_progress=False, **kwargs):
    # make sure we do not concurrently access the file
    with _p.lock(path, to_write=False):
        if path.lower().endswith('.csv.zip'):
            with _zf.ZipFile(path, mode='r') as myzip:
                if show_progress:
                    bar = tqdm(total=3, unit='step')

                filename = _p.basename(path)[:-4]

                # extract 'index_col' and 'dtype' from kwargs
                index_col = kwargs.pop('index_col', None)
                dtype = kwargs.pop('dtype', None)

                # load the metadata
                with myzip.open(filename[:-4]+'.meta', mode='r') as f:
                    meta = _js.load(f)
                if show_progress:
                    bar.update()

                # From now on, meta takes priority over dtype. We will ignore dtype.
                kwargs['dtype'] = 'object'

                # update index_col if it does not exist and meta has it
                if index_col is None and len(meta['index_names']) > 0:
                    index_col = meta['index_names']

                # Use pandas to load
                with myzip.open(filename, mode='r', force_zip64=True) as f:
                    df = _pd.read_csv(f, quoting=_csv.QUOTE_NONNUMERIC, **kwargs)
                if show_progress:
                    bar.update()
        else:
            # If '.meta' file exists, assume our format and therefore use dask to read. Otherwise, assume general csv file and use pandas to read.
            path2 = path[:-4]+'.meta'
            if not _p.exists(path2):  # no meta
                if show_progress:
                    bar = tqdm(total=2, unit='step')
                df = None
                if _p.getsize(path) >= (1 << 25):  # >= 32MB?
                    try:  # try to load in parallel using dask
                        df = _dd.read_csv(path, quoting=_csv.QUOTE_NONNUMERIC, **kwargs).compute()
                    except:
                        pass
                if show_progress:
                    bar.update()
                if not df:
                    df = _pd.read_csv(path, quoting=_csv.QUOTE_NONNUMERIC, **kwargs)
                if show_progress:
                    bar.update()
                    bar.close()
                return df

            if show_progress:
                bar = tqdm(total=3, unit='step')

            # extract 'index_col' and 'dtype' from kwargs
            index_col = kwargs.pop('index_col', None)
            dtype = kwargs.pop('dtype', None)

            # load the metadata
            meta = _js.load(open(path2, 'rt')) if _p.exists(path2) else None
            if show_progress:
                bar.update()

            # From now on, meta takes priority over dtype. We will ignore dtype.
            kwargs['dtype'] = 'object'

            # update index_col if it does not exist and meta has it
            if index_col is None and len(meta['index_names']) > 0:
                index_col = meta['index_names']

            # Try dask. If it fails, try pandas.
            try:
                df = _dd.read_csv(
                    path, quoting=_csv.QUOTE_NONNUMERIC, **kwargs).compute()
            except:
                df = _pd.read_csv(path, quoting=_csv.QUOTE_NONNUMERIC, **kwargs)
            if show_progress:
                bar.update()

        # adjust the returning dataframe based on the given meta
        s = meta['columns']
        for x in s:
            y = s[x]
            if y == 'datetime64[ns]':
                df[x] = _pd.to_datetime(df[x])
            elif isinstance(y, list) and y[0] == 'category':
                df[x] = df[x].astype('category', categories=y[1], ordered=y[2])
            elif y == 'int64':
                df[x] = df[x].astype(_np.int64)
            elif y == 'uint8':
                df[x] = df[x].astype(_np.uint8)
            elif y == 'float64':
                df[x] = df[x].astype(_np.float64)
            elif y == 'bool':
                # dd is very strict at reading a csv. It may read True as 'True' and False as 'False'.
                df[x] = df[x].replace('True', True).replace(
                    'False', False).astype(_np.bool)
            elif y == 'object':
                pass
            else:
                raise OSError("Unknown dtype for conversion {}".format(y))

        # set the index_col if it exists
        if index_col is not None and len(index_col) > 0:
            df = df.set_index(index_col, drop=True)

        if show_progress:
            bar.update()
            bar.close()

        return df


read_csv.__doc__ = '''Read a CSV file or a CSV-zipped file into a pandas.DataFrame, passing all arguments to :func:`dask.dataframe.read_csv` or :func:`pandas.read_csv` depending on how large the file is. Keyword argument 'show_progress' tells whether to show a progress bar in the terminal.''' + _dd.read_csv.__doc__


def to_csv(df, path, index='auto', file_mode=0o664, show_progress=False, **kwargs):

    if show_progress:
        bar = tqdm(total=3, unit='step')

    if index=='auto':
        index = bool(df.index.name)

    # make sure we do not concurrenly access the file
    with _p.lock(path, to_write=True):
        if path.lower().endswith('.csv.zip'):
            # write the csv file
            path2 = path+'.tmp.zip'
            dirpath = _p.dirname(path)
            if dirpath:
                _p.make_dirs(dirpath)
            if not _p.exists(dirpath):
                _t.sleep(1)

            with _zf.ZipFile(path2, mode='w') as myzip:
                filename = _p.basename(path)[:-4]
                with myzip.open(filename, mode='w', force_zip64=True) as f: # csv
                    data = df.to_csv(None, index=index, quoting=_csv.QUOTE_NONNUMERIC, **kwargs)
                    f.write(data.encode())
                if show_progress:
                    bar.update()
                with myzip.open(filename[:-4]+'.meta', mode='w') as f: # meta
                    data = _js.dumps(metadata(df))
                    f.write(data.encode())
                if show_progress:
                    bar.update()
                res = None
            if file_mode:  # chmod
                _p.chmod(path2, file_mode)
        else:
            # write the csv file
            path2 = path+'.tmp.csv'
            dirpath = _p.dirname(path)
            if dirpath:
                _p.make_dirs(dirpath)
            if not _p.exists(dirpath):
                _t.sleep(1)
            res = df.to_csv(path2, index=index, quoting=_csv.QUOTE_NONNUMERIC, **kwargs)
            if file_mode:  # chmod
                _p.chmod(path2, file_mode)
            if show_progress:
                bar.update()

            # write the meta file
            path3 = path[:-4]+'.meta'
            _js.dump(metadata(df), open(path3, 'wt'))
            try:
                if file_mode:  # chmod
                    _p.chmod(path3, file_mode)
            except PermissionError:
                pass # for now
            if show_progress:
                bar.update()

        _p.remove(path)
        if _p.exists(path) or not _p.exists(path2):
            _t.sleep(1)
        _p.rename(path2, path)
        if show_progress:
            bar.update()
            bar.close()
        return res

to_csv.__doc__ = '''Write DataFrame to a comma-separated values (.csv) file or a CSV-zipped (.csv.zip) file. If keyword 'index' is 'auto' (default), the index column is written if and only if it has a name. Keyword 'file_mode' specifies the file mode when writing (passed directly to os.chmod if not None), and the remaining arguments and keywords are passed directly to :func:`DataFrame.to_csv`. Keyword argument 'show_progress' tells whether to show a progress bar in the terminal.\n''' + _pd.DataFrame.to_csv.__doc__
