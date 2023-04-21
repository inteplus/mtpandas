#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from packaging import version as pv
from mt.pandas.version import version

import platform

py_ver = platform.python_version()

if pv.parse(py_ver) < pv.parse("3.7"):
    dependencies = ["dataclasses"]
else:
    dependencies = []

setup(
    name="mtpandas",
    version=version,
    description="MT's extra modules for pandas.",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    install_requires=dependencies
    + [
        "halo",  # for showing some spinners
        "pandas>=1.3",  # for dataframes, and we need custom dtypes
        #'h5py>=3', # for pdh5 file format. Lazy import because TX2 may not need it.
        #'pyarrow', # for converting to/from parquet. But TX2 doesn't need pyarrow.
        "mtbase>=4.4",  # to have module 'dataframe_processing' migrated from mtbase to here
        "mtopencv>=1.9",  # Minh-Tri's OpenCV wrapper so we can use cv.Image
        "tqdm",  # for drawing progress bars
        "tabview",  # for viewing spreadsheet-like files
    ],
    python_requires=">=3.6",  # we still have to support TX1 modules
    scripts=[
        "scripts/dfconvert",
        "scripts/dfview",
    ],
    url="https://github.com/inteplus/mtpandas",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.pandas/mt.pandas.html",
        "Source Code": "https://github.com/inteplus/mtpandas",
    },
)
