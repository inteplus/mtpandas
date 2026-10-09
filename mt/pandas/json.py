'''Custom json accessor for pandas.

Importing :mod:`mt.pandas` registers the accessor as ``Series.json``.
'''


import numpy as _np
import json
import pandas.api.extensions as _pae


__all__ = ['JsonAccessor']


@_pae.register_series_accessor("json")
class JsonAccessor:
    '''Accessor for json fields, available as ``Series.json``.

    A json field is a series whose items are json-like Python objects (dict, list, str, int,
    float, bool) or None. Every property below returns a new series with the same index as the
    original one.

    Examples
    --------
    >>> import pandas as pd
    >>> import mt.pandas
    >>> s = pd.Series([{"a": 1}, [1, 2], None])
    >>> s.json.to_str
    0    {"a": 1}
    1      [1, 2]
    2        None
    dtype: object
    >>> s.json.to_str.json.from_str.tolist()
    [{'a': 1}, [1, 2], None]
    '''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def valid(self):
        '''Returns a boolean series telling which item is a json-like object.

        An item is accepted if it is a dict, list, str, int, float or bool. None is not accepted.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([{"a": 1}, [1, 2], "x", None]).json.valid.tolist()
        [True, True, True, False]
        '''
        return self._obj.apply(lambda x: isinstance(x, (dict, list, str, int, float, bool)))

    @property
    def valid_str(self):
        '''Returns a boolean series telling which item is a string.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([{"a": 1}, "x"]).json.valid_str.tolist()
        [False, True]
        '''
        return self._obj.apply(lambda x: isinstance(x, str))

    @property
    def valid_sequence(self):
        '''Returns a boolean series telling which item is a dict, a list or a string.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([{"a": 1}, [1, 2], "x", None, 3]).json.valid_sequence.tolist()
        [True, True, True, False, False]
        '''
        return self._obj.apply(lambda x: isinstance(x, (dict, list, str)))

    @property
    def to_ndarray(self):
        '''Converts every item into a :class:`numpy.ndarray`. None is kept as None.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([[1, 2], None]).json.to_ndarray.tolist()
        [array([1, 2]), None]
        '''
        return self._obj.apply(lambda x: None if x is None else _np.array(x))

    @property
    def from_ndarray(self):
        '''Converts every ndarray item into a (nested) list. None is kept as None.

        Raises
        ------
        AttributeError
            if an item is neither None nor a :class:`numpy.ndarray`

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([np.array([1, 2]), None]).json.from_ndarray.tolist()
        [[1, 2], None]
        '''
        def func(x):
            if x is None:
                return None
            if not isinstance(x, _np.ndarray):
                raise AttributeError(
                    f"Expected an ndarray, received an object of type '{type(x)}' instead."
                )
            return x.tolist()
        return self._obj.apply(func)
        

    @property
    def to_str(self):
        '''Serialises every item into a json string using :func:`json.dumps`. None is kept as None.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([{"a": 1}, None]).json.to_str.tolist()
        ['{"a": 1}', None]
        '''
        return self._obj.apply(lambda x: None if x is None else json.dumps(x))

    @property
    def from_str(self):
        '''Parses every json string item into an object using :func:`json.loads`.

        None is kept as None. This is the inverse of :attr:`to_str`.

        Raises
        ------
        AttributeError
            if an item is neither None nor a string

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series(['{"a": 1}', '[1, 2]']).json.from_str.tolist()
        [{'a': 1}, [1, 2]]
        '''
        def func(x):
            if x is None:
                return None
            if not isinstance(x, str):
                raise AttributeError(
                    f"Expected a string, received an object of type '{type(x)}' instead."
                )
            return json.loads(x)
        return self._obj.apply(func)
