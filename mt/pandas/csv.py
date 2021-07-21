import json
import pandas as pd
from tqdm import tqdm
from halo import Halo
import csv as _csv
from mt import np
import mt.base.path as _p
from mt.base.with_utils import dummy_scope, join_scopes
import json
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

    def postprocess(df):
        # special treatment of fields introduced by function dfpack()
        for key in df:
            if key.endswith('_df_nd_ravel'):
                has_ndarray = True
                break
        else:
            has_ndarray = False
        if has_ndarray:
            df = df.copy() # to avoid generating a warning
            fromlist = lambda x: None if x is None else np.array(json.loads(x))
            for key in df:
                if key.endswith('_df_nd_ravel'):
                    df[key] = df[key].apply(fromlist)
                elif key.endswith('_df_nd_shape'):
                    df[key] = df[key].apply(fromlist)
        return df

    def process(path, fp1, fp2, show_progress=False, **kwargs):

        # do read
        if show_progress:
            bar = tqdm(unit='row')
        dfs = []
        for df in pd.read_csv(fp1, quoting=_csv.QUOTE_NONNUMERIC, chunksize=1024, **kwargs):
            dfs.append(df)
            if show_progress:
                bar.update(len(df))
        df = pd.concat(dfs, sort=False)
        if show_progress:
            bar.close()

        # If '.meta' file exists, assume general csv file and use pandas to read.
        if fp2 is None: # no meta
            return postprocess(df)

        spinner = Halo(text="dfloading '{}'".format(path), spinner='dots') if show_progress else dummy_scope
        with spinner:
            try:
                # extract 'index_col' and 'dtype' from kwargs
                index_col = kwargs.pop('index_col', None)
                dtype = kwargs.pop('dtype', None)

                # load the metadata
                if show_progress:
                    spinner.text = "loading the metadata"
                meta = None if fp2 is None else json.load(fp2)

                if meta is None:
                    if show_progress:
                        spinner.succeed("dfloaded '{}' with no metadata".format(path))
                    return postprocess(df)

                # From now on, meta takes priority over dtype. We will ignore dtype.
                kwargs['dtype'] = 'object'

                # update index_col if it does not exist and meta has it
                if index_col is None and len(meta['index_names']) > 0:
                    index_col = meta['index_names']

                # adjust the returning dataframe based on the given meta
                s = meta['columns']
                for x in s:
                    if show_progress:
                        spinner.text = "checking column '{}'".format(x)
                    y = s[x]
                    if y == 'datetime64[ns]':
                        df[x] = pd.to_datetime(df[x])
                    elif isinstance(y, list) and y[0] == 'category':
                        cat_dtype = pd.api.types.CategoricalDtype(categories=y[1], ordered=y[2])
                        df[x] = df[x].astype(cat_dtype)
                    elif y == 'int64':
                        df[x] = df[x].astype(np.int64)
                    elif y == 'uint8':
                        df[x] = df[x].astype(np.uint8)
                    elif y == 'float64':
                        df[x] = df[x].astype(np.float64)
                    elif y == 'bool':
                        # dd is very strict at reading a csv. It may read True as 'True' and False as 'False'.
                        df[x] = df[x].replace('True', True).replace(
                            'False', False).astype(np.bool)
                    elif y == 'object':
                        pass
                    else:
                        raise OSError("Unknown dtype for conversion {}".format(y))

                # set the index_col if it exists
                if index_col is not None and len(index_col) > 0:
                    df = df.set_index(index_col, drop=True)

                if show_progress:
                    spinner.succeed("dfloaded '{}'".format(path))
            except:
                if show_progress:
                    spinner.succeed("failed to dfload '{}'".format(path))
                raise

        return postprocess(df)

    # make sure we do not concurrently access the file
    with _p.lock(path, to_write=False):
        if path.lower().endswith('.csv.zip'):
            with _zf.ZipFile(path, mode='r') as myzip:
                filename = _p.basename(path)[:-4]
                fp1 = myzip.open(filename, mode='r', force_zip64=True)
                meta_filename = filename[:-4]+'.meta'
                if meta_filename in myzip.namelist():
                    fp2 = myzip.open(meta_filename, mode='r')
                else:
                    fp2 = None
                return process(path, fp1, fp2, show_progress=show_progress, **kwargs)
        else:
            fp1 = path
            meta_filepath = _p.basename(path)[:-4]+'.meta'
            if _p.exists(meta_filepath):
                fp2 = open(meta_filepath, 'rt')
            else:
                fp2 = None
            return process(path, fp1, fp2, show_progress=show_progress, **kwargs)

read_csv.__doc__ = '''Read a CSV file or a CSV-zipped file into a pandas.DataFrame, passing all arguments to :func:`pandas.read_csv`. Keyword argument 'show_progress' tells whether to show progress in the terminal.''' + pd.read_csv.__doc__


def to_csv(df, path, index='auto', file_mode=0o664, show_progress=False, **kwargs):

    # special treatment of fields introduced by function dfpack()
    for key in df:
        if key.endswith('_df_nd_ravel'):
            has_ndarray = True
            break
    else:
        has_ndarray = False
    if has_ndarray:
        df = df.copy() # to avoid generating a warning
        tolist = lambda x: None if x is None else json.dumps(x.tolist())
        for key in df:
            if key.endswith('_df_nd_ravel'):
                df[key] = df[key].apply(tolist)
            elif key.endswith('_df_nd_shape'):
                df[key] = df[key].apply(tolist)

    spinner = Halo(text="dfsaving '{}'".format(path), spinner='dots') if show_progress else dummy_scope
    with spinner:
        try:
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
                            spinner.text = "saved CSV content"
                        with myzip.open(filename[:-4]+'.meta', mode='w') as f: # meta
                            data = json.dumps(metadata(df))
                            f.write(data.encode())
                        if show_progress:
                            spinner.text = "saved metadata"
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
                        spinner.text = "saved CSV content"

                    # write the meta file
                    path3 = path[:-4]+'.meta'
                    json.dump(metadata(df), open(path3, 'wt'))
                    try:
                        if file_mode:  # chmod
                            _p.chmod(path3, file_mode)
                    except PermissionError:
                        pass # for now
                    if show_progress:
                        spinner.text = "saved metadata"

                _p.remove(path)
                if _p.exists(path) or not _p.exists(path2):
                    _t.sleep(1)
                _p.rename(path2, path)

                if isinstance(spinner, Halo):
                    spinner.succeed("dfsaved '{}'".format(path))

                return res

        except:
            if isinstance(spinner, Halo):
                spinner.succeed("failed to dfsave '{}'".format(path))
            raise

to_csv.__doc__ = '''Write DataFrame to a comma-separated values (.csv) file or a CSV-zipped (.csv.zip) file. If keyword 'index' is 'auto' (default), the index column is written if and only if it has a name. Keyword 'file_mode' specifies the file mode when writing (passed directly to os.chmod if not None), and the remaining arguments and keywords are passed directly to :func:`DataFrame.to_csv`. Keyword argument 'show_progress' tells whether to show progress in the terminal.\n''' + pd.DataFrame.to_csv.__doc__
