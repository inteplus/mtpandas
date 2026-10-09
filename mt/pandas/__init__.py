"""Extra functions and pandas accessors for working with dataframes and series.

Importing the package registers the ``Series.json``, ``Series.ndarray`` and ``Series.word``
accessors and exposes the public functions of its submodules: file I/O (:func:`dfload`,
:func:`dfsave`, ...), series and dataframe helpers, dataclass columns, :func:`get_dftype` and
:class:`WranglingFrame`. For pandas plus these extras in one namespace, use ``from mt import pd``.
"""

from .version import version as __version__
from .custom_accessors import *
from .series import *
from .dataframe import *
from .dataclass import *
from .dftype import *
from .convert import *
from .wrangling_frame import *
