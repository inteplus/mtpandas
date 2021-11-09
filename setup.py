#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.pandas.version import version

setup(
    name='mtpandas',
    version=version,
    description="MT's extra modules for pandas.",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=['mt.*']),
    install_requires=[
        'halo', # for showing some spinners
        'pandas>=1.2.3',  # for dataframes, and we need custom dtypes
        'pandarallel', # for parallel apply in pandas
        'h5py>=3', # for pdh5 file format
        #'pyarrow', # for converting to/from parquet. But TX2 doesn't need pyarrow.
        'mtbase>=2.5',  # Minh-Tri's base modules for logging and multi-threading
        'mtopencv>=1.1', # Minh-Tri's OpenCV wrapper so we can use cv.Image
        'tqdm', # for drawing progress bars
        'tabview', # for viewing spreadsheet-like files
    ],
    scripts=[
        'scripts/dfconvert',
        'scripts/dfview',
    ],
    url='https://github.com/inteplus/mtpandas',
    project_urls={
        'Documentation': 'https://mtdoc.readthedocs.io/en/latest/mt.pandas/mt.pandas.html',
        'Source Code': 'https://github.com/inteplus/mtpandas',
    }
)
