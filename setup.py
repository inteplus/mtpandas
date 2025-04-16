#!/usr/bin/env python3

import os
import platform
from setuptools import setup, find_namespace_packages
from packaging import version as pv

py_ver = platform.python_version()

if pv.parse(py_ver) < pv.parse("3.7"):
    dependencies = ["dataclasses"]
else:
    dependencies = []

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mtpandas",
    description="MT's extra modules for pandas.",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    install_requires=dependencies
    + [
        "pandas>=1.5",  # for dataframes, and we need custom dtypes
        "pandas_parallel_apply",  # for parallel_apply functions
        #'h5py>=3', # for pdh5 file format. Lazy import because TX2 may not need it.
        #'pyarrow', # for converting to/from parquet. But TX2 doesn't need pyarrow.
        "mtbase>=4.30.24",  # to have mt.traceback.LogicError
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
    setup_requires=["setuptools-git-versioning<2"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
    license="MIT",
    license_files=["LICENSE"],
)
