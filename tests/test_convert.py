import os
import tempfile
import unittest

import pandas as pd

import mt.pandas
from mt.pandas import dfload, dfsave

assert mt.pandas.__file__.startswith("/home/minhtri/gitcentre/mtpandas"), mt.pandas.__file__

try:
    import pyarrow  # noqa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


def make_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y,z", "w"], "c": [0.5, 1.5, 2.5]})
    df.index.name = "id"
    return df


class TestRoundTrip(unittest.TestCase):
    def check(self, name):
        df = make_df()
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, name)
            dfsave(df, fp)
            pd.testing.assert_frame_equal(dfload(fp), df)

    def test_csv(self):
        self.check("df.csv")

    def test_csv_zip(self):
        self.check("df.csv.zip")

    @unittest.skipUnless(HAS_ARROW, "pyarrow not installed")
    def test_parquet(self):
        self.check("df.parquet")

    def test_csv_meta_next_to_file_not_cwd(self):
        df = make_df()
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            fp = os.path.join(d, "df.csv")
            dfsave(df, fp)
            self.assertTrue(os.path.exists(os.path.join(d, "df.meta")))
            old = os.getcwd()
            os.chdir(other)
            try:
                pd.testing.assert_frame_equal(dfload(fp), df)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
