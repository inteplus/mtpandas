'''Custom ndarray accessor for pandas.

Importing :mod:`mt.pandas` registers the accessor as ``Series.ndarray``.
'''


import numpy as _np
import pandas.api.extensions as _pae


__all__ = ['NdarrayAccessor']


@_pae.register_series_accessor("ndarray")
class NdarrayAccessor:
    '''Accessor for ndarray fields, available as ``Series.ndarray``.

    An ndarray field is a series whose items are :class:`numpy.ndarray` instances or None.
    '''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def valid(self):
        '''Returns a boolean series telling which item is a :class:`numpy.ndarray`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([np.zeros(2), None, [1]]).ndarray.valid.tolist()
        [True, False, False]
        '''
        return self._obj.apply(lambda x: isinstance(x, _np.ndarray))

    @property
    def to_json(self):
        '''Returns a series of (nested) lists converted from the ndarrays using ``tolist()``.

        None is kept as None. Any other non-ndarray item raises an :class:`AttributeError`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series([np.arange(3), None]).ndarray.to_json.tolist()
        [[0, 1, 2], None]
        '''
        return self._obj.apply(lambda x: None if x is None else x.tolist())
