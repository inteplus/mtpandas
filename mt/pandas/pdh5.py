'''Loading and saving to column-based pdh5 format.'''

from typing import Optional

import os
import json
from contextlib import nullcontext
import pandas as pd
import h5py
from halo import Halo

from mt import np, cv
from mt.base.str import text_filename
from mt.base.path import rename
from .dftype import get_dftype


__all__ = ['save_pdh5', 'load_pdh5', 'Pdh5Cell']


def isnull(obj):
    if obj is None or obj is pd.NaT:
        return True
    if isinstance(obj, float) and np.isnan(obj):
        return True
    return False


def load_special_cell(grp, key, dftype):
    if dftype == 'ndarray':
        return grp[key][:]
    if dftype == 'SparseNdarray':
        grp2 = grp.require_group(key)
        dense_shape = tuple(json.loads(grp2.attrs['dense_shape']))
        values = grp2['values'][:]
        indices = grp2['indices'][:]
        return np.SparseNdarray(values, indices, dense_shape)
    if dftype == 'Image':
        grp2 = grp.require_group(key)
        pixel_format = grp2.attrs['pixel_format']
        meta = json.loads(grp2.attrs['meta'])
        image = grp2['image'][:]
        return cv.Image(image, pixel_format=pixel_format, meta=meta)
    raise ValueError("Unknown dftype while loading cells: '{}'.".format(dftype))


class Pdh5Column:
    '''A read-only column of a pdh5 file.'''
    def __init__(self, filepath: str, col_id: str):
        self.filepath = filepath
        self.col_id = col_id
        self.col = None
        self.dftype = None
        self.loaded = False

    def get_item(self, row_id: int):
        if not self.loaded:
            f = h5py.File(self.filepath, mode='r')
            columns = json.loads(f.attrs['columns'])
            self.dftype = columns[self.col_id]
            key = 'column_'+text_filename(self.col_id)
            if self.dftype != 'none':
                self.col = f[key]
            self.loaded = True

        if self.dftype == 'none':
            return None
        if self.dftype == 'json':
            x = self.col[row_id]
            return None if x == b'' else json.loads(x)
        if self.dftype in ('ndarray', 'Image', 'SparseNdarray'):
            key = str(row_id)
            if not key in self.col:
                return None
            return load_special_cell(self.col, key, self.dftype)


class Pdh5Cell:
    '''A read-only cell of a pdh5 column.'''

    def __init__(self, col: Pdh5Column, row_id: int):
        self.col = col
        self.row_id = row_id
        self._value = None
        self.loaded = False

    @property
    def value(self):
        if not self.loaded:
            self._value = self.col.get_item(self.row_id)
            self.loaded = True
        return self._value


def save_pdh5_index(f: h5py.File, df: pd.DataFrame, spinner=None):
    f.attrs['format'] = 'pdh5'
    f.attrs['version'] = '1.0'
    size = len(df)
    f.attrs['size'] = size

    index = df.index
    grp = f.create_group("index")

    if spinner is not None:
        spinner.text = 'saving index of type {}'.format(type(index))

    if isinstance(index, pd.RangeIndex):
        grp.attrs['type'] = 'RangeIndex'
        if index.start is not None:
            grp.attrs['start'] = index.start
        if index.stop is not None:
            grp.attrs['stop'] = index.stop
        if index.step is not None:
            grp.attrs['step'] = index.step
        if index.name is not None:
            grp.attrs['name'] = index.name
    elif isinstance(index, (pd.Int64Index, pd.UInt64Index, pd.Float64Index)):
        grp.attrs['type'] = type(index).__name__
        if index.name is not None:
            grp.attrs['name'] = index.name
        data = grp.create_dataset(name='values', data=index.values, compression='gzip')
    else:
        raise ValueError("Unsupported index type '{}'.".format(type(index)))


def save_pdh5_columns(f: h5py.File, df: pd.DataFrame, spinner=None):
    columns = {x: get_dftype(df[x]) for x in df.columns}
    f.attrs['columns'] = json.dumps(columns)

    for column in columns:
        if spinner is not None:
            spinner.text = "saving column '{}'".format(column)
        key = 'column_'+text_filename(column)
        dftype = columns[column]
        if dftype == 'none':
            pass
        elif dftype == 'str':
            data = df[column].apply(lambda x: '\0' if isnull(x) else x).to_numpy().astype('S')
            f.create_dataset(key, data=data, compression='gzip')
        elif dftype in ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'int64', 'uint64', 'float64'):
            data = df[column].to_numpy()
            f.create_dataset(key, data=data, compression='gzip')
        elif dftype == 'json':
            data = df[column].apply(lambda x: '\0' if isnull(x) else json.dumps(x)).to_numpy().astype('S')
            f.create_dataset(key, data=data, compression='gzip')
        elif dftype in ('Timestamp', 'Timedelta'):
            data = df[column].apply(lambda x: '\0' if isnull(x) else str(x)).to_numpy().astype('S')
            f.create_dataset(key, data=data, compression='gzip')
        elif dftype in ('ndarray', 'Image', 'SparseNdarray'):
            data = df[column].tolist()
            grp = f.create_group(key)
            for i, item in enumerate(data):
                if isnull(item):
                    continue
                key = str(i)
                if dftype == 'ndarray':
                    grp.create_dataset(key, data=item, compression='gzip')
                elif dftype == 'SparseNdarray':
                    grp2 = grp.create_group(key)
                    grp2.attrs['dense_shape'] = json.dumps(item.dense_shape)
                    grp2.create_dataset('values', data=item.values, compression='gzip')
                    grp2.create_dataset('indices', data=item.indices, compression='gzip')
                elif dftype == 'Image':
                    grp2 = grp.create_group(key)
                    grp2.attrs['pixel_format'] = item.pixel_format
                    grp2.attrs['meta'] = json.dumps(item.meta)
                    grp2.create_dataset('image', data=item.image, compression='gzip')
        else:
            data = df[column].apply(lambda x: type(x)).unique()
            raise ValueError("Unable to save column '{}' with type list '{}'.".format(column, data))


def save_pdh5(filepath: str, df: pd.DataFrame, file_mode: Optional[int] = 0o664, show_progress: bool = False):
    '''Saves a dataframe into a .pdh5 file.

    Parameters
    ----------
    filepath : str
        path to the file to be written to
    df : pandas.DataFrame
        the dataframe to write from
    file_mode : int, optional
        file mode of the newly written file
    show_progress : bool
        show a progress spinner in the terminal
    '''
    if show_progress:
        spinner = Halo("dfsaving '{}'".format(filepath), spinner='dots')
        scope = spinner
    else:
        spinner = None
        scope = nullcontext()
    try:
        filepath2 = filepath+'.mttmp'
        with scope, h5py.File(filepath2, 'w') as f:
            save_pdh5_index(f, df, spinner=spinner)
            save_pdh5_columns(f, df, spinner=spinner)
        if file_mode is not None:  # chmod
            os.chmod(filepath2, file_mode)
        rename(filepath2, filepath, overwrite=True)
        if show_progress:
            spinner.succeed("dfsaved '{}'".format(filepath))
    except:
        if show_progress:
            spinner.fail("failed to dfsave '{}'".format(filepath))
        raise


def load_pdh5_index(f: h5py.File, spinner=None):
    if f.attrs['format'] != 'pdh5':
        raise ValueError("Input file does not have 'pdh5' format.")
    size = f.attrs['size']

    grp = f.require_group("index")

    index_type = grp.attrs['type']
    if spinner is not None:
        spinner.text = 'loading index of type {}'.format(index_type)

    if index_type == 'RangeIndex':
        start = grp.attrs.get('start', None)
        stop = grp.attrs.get('stop', None)
        step = grp.attrs.get('step', None)
        name = grp.attrs.get('name', None)
        index = pd.RangeIndex(start=start, stop=stop, step=step, name=name)
    elif index_type in ('Int64Index', 'UInt64Index', 'Float64Index'):
        name = grp.attrs.get('name', None)
        values = grp['values'][:]
        index = getattr(pd, index_type)(data=values, name=name)
    else:
        raise ValueError("Unsupported index type '{}'.".format(type(index)))

    return pd.DataFrame(index=index)


def load_pdh5_columns(f: h5py.File, df: pd.DataFrame, spinner=None, file_read_delayed: bool = False):
    columns = json.loads(f.attrs['columns'])

    for column in columns:
        if spinner is not None:
            spinner.text = "loading column '{}'".format(column)
        key = 'column_'+text_filename(column)
        dftype = columns[column]
        if dftype == 'none':
            df[column] = None
        elif dftype == 'str':
            df[column] = f[key][:]
            df[column] = df[column].apply(lambda x: None if x == b'' else x.decode())
        elif dftype in ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'int64', 'uint64', 'float64'):
            df[column] = f[key][:]
        elif dftype == 'json':
            if file_read_delayed:
                col = Pdh5Column(f.filename, column)
                df[column] = [Pdh5Cell(col, i) for i in range(len(df.index))]
            else:
                df[column] = f[key][:]
                df[column] = df[column].apply(lambda x: None if x == b'' else json.loads(x))
        elif dftype == 'Timestamp':
            df[column] = f[key][:]
            df[column] = df[column].apply(lambda x: pd.NaT if x == b'' else pd.Timestamp(x.decode()))
        elif dftype == 'Timedelta':
            df[column] = f[key][:]
            df[column] = df[column].apply(lambda x: pd.NaT if x == b'' else pd.Timedelta(x.decode()))
        elif dftype in ('ndarray', 'Image', 'SparseNdarray'):
            data = [None]*len(df.index)
            grp = f.require_group(key)
            if file_read_delayed:
                col = Pdh5Column(f.filename, column)
            for key in grp.keys():
                i = int(key)
                data[i] = Pdh5Cell(col, i) if file_read_delayed else load_special_cell(grep, key, dftype)
            df[column] = data
        else:
            raise ValueError("Unable to load column '{}' with dftype '{}'.".format(column, dftype))


def load_pdh5(filepath: str, show_progress: bool = False, file_read_delayed: bool = False):
    '''Loads the dataframe of a .pdh5 file.

    Parameters
    ----------
    filepath : str
        path to the file to be read from
    show_progress : bool
        show a progress spinner in the terminal
    file_read_delayed: bool
        If True, columns of dftype 'json', 'ndarray', 'Image' and 'SparseNdarray' are proxied for
        reading later, returning cells are instances of :class:`Pdh5Cell` instead. If False, these
        columns are read thoroughly, which can be slow.

    Returns
    -------
    df : pandas.DataFrame
        the loaded dataframe
    '''
    if show_progress:
        spinner = Halo("dfloading '{}'".format(filepath), spinner='dots')
        scope = spinner
    else:
        spinner = None
        scope = nullcontext()
    try:
        with scope, h5py.File(filepath, 'r') as f:
            df = load_pdh5_index(f, spinner=spinner)
            load_pdh5_columns(f, df, spinner=spinner, file_read_delayed=file_read_delayed)
        if show_progress:
            spinner.succeed("dfloaded '{}'".format(filepath))
        return df
    except:
        if show_progress:
            spinner.fail("failed to load '{}'".format(filepath))
        raise
