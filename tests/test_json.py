import unittest

import pandas as pd

import mt.pandas
import mt.pandas.json

assert mt.pandas.__file__.startswith("/home/minhtri/gitcentre/mtpandas"), mt.pandas.__file__


class TestJsonAccessor(unittest.TestCase):
    def test_valid_sequence(self):
        s = pd.Series([{"a": 1}, [1, 2], "x", None, 3])
        self.assertEqual(s.json.valid_sequence.tolist(), [True, True, True, False, False])

    def test_valid_sequence_empty(self):
        self.assertEqual(pd.Series([], dtype=object).json.valid_sequence.tolist(), [])

    def test_str_round_trip(self):
        s = pd.Series([{"a": 1}, [1, 2], None])
        self.assertEqual(s.json.to_str.json.from_str.tolist(), [{"a": 1}, [1, 2], None])


if __name__ == "__main__":
    unittest.main()
