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
        'pandas>=0.23.0',  # for dataframes, and we need custom dtypes
        'dask[dataframe]',  # for reading chunks of CSV files in parallel
        'mtbase>=0.4.0',  # Minh-Tri's base modules for logging and multi-threading
    ],
    url='https://github.com/inteplus/mtpandas',
    project_urls={
        'Documentation': 'https://mtpandas.readthedocs.io/en/stable/',
        'Source Code': 'https://github.com/inteplus/mtpandas',
    }
)
