'''Custom json accessor for pandas.'''


import numpy as _np
import json
import pandas.api.extensions as _pae


__all__ = ['JsonAccessor']


@_pae.register_series_accessor("json")
class JsonAccessor:
    '''Accessor for json fields.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def valid(self):
        '''Returns which item is a valid json object.'''
        return self._obj.apply(lambda x: isinstance(x, (dict, list, str, int, float, bool)))

    @property
    def valid_str(self):
        '''Returns which item is a valid string.'''
        return self._obj.apply(lambda x: isinstance(x, str))

    @property
    def valid_sequence(self):
        '''Returns which item is a valid sequence.'''
        return self._obj.apply(lambda x: isinstance(dict, list, str))

    @property
    def to_ndarray(self):
        '''Converts into an ndarray series.'''
        return self._obj.apply(lambda x: None if x is None else _np.array(x))

    @property
    def from_ndarray(self):
        '''Converts from an ndarray.'''
        def func(x):
            if x is None:
                return None
            if not isinstance(x, _np.ndarray):
                raise AttributeError("Expected an ndarray, received an object of type '{}' instead.".format(type(x)))
            return x.tolist()
        return self._obj.apply(func)
        

    @property
    def to_str(self):
        '''Converts into a string series.'''
        return self._obj.apply(lambda x: None if x is None else json.dumps(x))

    @property
    def from_str(self):
        '''Converts from a string series.'''
        def func(x):
            if x is None:
                return None
            if not isinstance(x, str):
                raise AttributeError("Expected a string, received an object of type '{}' instead.".format(type(x)))
            return json.loads(x)
        return self._obj.apply(func)
