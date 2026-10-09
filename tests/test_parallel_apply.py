import unittest

import pandas as pd

import mt.pandas
from mt.pandas import parallel_groupby_apply
from mt.pandas.parallel_apply import DataFrameParallel, GroupByParallel


def add(d, k, scale=1):
    return (d["a"].sum() + k) * scale


class TestParallel(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"g": [0, 0, 1, 1], "a": [1, 2, 3, 4]})

    def test_groupby_honours_multithread(self):
        # a lambda cannot be pickled, so it only works if threads are really used
        dp = DataFrameParallel(self.df, 2, pbar=False, parallelism="multithread")
        gp = dp.groupby("g")
        self.assertIsInstance(gp, GroupByParallel)
        self.assertEqual(gp.parallelism, "multithread")
        self.assertFalse(gp.keep_original_indexes)
        self.assertEqual(gp.apply(lambda d: d["a"].sum()).tolist(), [3, 7])

    def test_groupby_apply_extra_args(self):
        res = parallel_groupby_apply(self.df, ["g"], add, func_args=(10,),
                                     func_kwds={"scale": 2}, n_cores=2,
                                     parallelism="multithread")
        self.assertEqual(res.tolist(), [26, 34])

    def test_groupby_apply_index_names(self):
        res = parallel_groupby_apply(self.df, ["g"], add, func_args=(0,), n_cores=2,
                                     parallelism="multithread")
        self.assertEqual(res.index.name, "g")
        df = pd.DataFrame({"g": [0, 0, 1, 1], "h": [1, 1, 1, 2], "a": [1, 2, 3, 4]})
        res = parallel_groupby_apply(df, ["g", "h"], add, func_args=(0,), n_cores=2,
                                     parallelism="multithread")
        self.assertEqual(list(res.index.names), ["g", "h"])
        self.assertEqual(res.tolist(), [3, 3, 4])


if __name__ == "__main__":
    unittest.main()
