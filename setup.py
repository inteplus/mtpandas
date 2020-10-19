#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.pandas.version import VERSION

setup(
    name='mtpandas',
    version=VERSION,
    description="MT's extra modules for pandas.",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=['mt.*']),
    install_requires=[
        'pandas>=1.0.3',  # for dataframes, and we need custom dtypes
        'pyarrow', # for converting to/from parquet
        'dask[dataframe]',  # for reading chunks of CSV files in parallel
        'mtbase>=0.6.0',  # Minh-Tri's base modules for logging and multi-threading
    ],
    scripts=[
        'scripts/dfconvert.py',
    ],
    url='https://github.com/inteplus/mtpandas',
    project_urls={
        'Documentation': 'https://mtdoc.readthedocs.io/en/latest/mt.pandas/mt.pandas.html',
        'Source Code': 'https://github.com/inteplus/mtpandas',
    }
)
