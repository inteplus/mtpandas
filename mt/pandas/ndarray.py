'''Custom ndarray accessor for pandas.'''


import numpy as _np
import pandas.api.extensions as _pae


__all__ = ['NdarrayAccessor']


@_pae.register_series_accessor("ndarray")
class NdarrayAccessor:
    '''Accessor for ndarray fields.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def valid(self):
        '''Returns which item is a valid ndarray'''
        return self._obj.apply(lambda x: isinstance(x, _np.ndarray))

    @property
    def to_json(self):
        '''Returns a series of json objects converted from the ndarrays'''
        return self._obj.apply(lambda x: None if x is None else x.tolist())
