'''Custom data types used in pandas. For now we implement json and ndarray.'''


import numpy as _np
import json as _js
import pandas.api.extensions as _pae

@_pae.register_extension_dtype
class JsonDtype(_pae.ExtensionDtype):
    '''Json as a custom pandas dtype.

    The json dtype declares None and anything that can be json-encoded as an json object.

    .. versionadded:: 0.1.3

    Notes
    -----
    This is still experimental. Besides, we need to wait for pandas to formalise ExtensionDtype first.
    '''

    @property
    def type(self):
        return object

    @property
    def name(self):
        return 'json'

    @classmethod
    def construct_array_type(cls):
        return JsonArray()

    @classmethod
    def construct_from_string(cls, string):
        if string == 'json':
            return cls()
        else:
            raise TypeError("Cannot construct a '{cls.__name__}' from '{string}'")

class JsonArray(_pae.ExtensionArray):
    '''Pandas array of ndarray.

    .. versionadded:: 0.1.3

    Notes
    -----
    This is still experimental. Besides, we need to wait for pandas to formalise ExtensionArray first.
    '''

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def dtype(self) -> ExtensionDtype:
        """
        An instance of 'JsonDtype'.
        """
        return JsonDtype()

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        raise AbstractMethodError(self)

    @property
    def one_plus_one(self) -> tow:
        raise NotImplementedError

    # MT-TODO: work in progress
