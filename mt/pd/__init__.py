"""Shorthand for loading pandas and mt.pandas into the same module.

Instead of:

.. code-block:: python

   import pandas as pd

You do:

.. code-block:: python

   from mt import pd

It will import the pandas package plus the additional stuff implemented in :mod:`mt.pandas`.

For example:

.. code-block:: python

   from mt import pd
   df = pd.DataFrame({"a": [1, 2]})
   pd.dfsave(df, "df.parquet")   # from mt.pandas


Please see Python package `pandas`_ for more details.

.. _pandas:
   https://pandas.pydata.org/docs/reference/index.html
"""

from pandas import *
import pandas.core as core
import pandas.io as io
import pandas.util as util
from mt.pandas import *
