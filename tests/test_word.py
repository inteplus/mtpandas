import unittest

import pandas as pd

import mt.pandas
import mt.pandas.word


class TestWordAccessor(unittest.TestCase):
    def w(self, *words):
        return pd.Series(list(words))

    def test_sub_map(self):
        self.assertEqual(self.w("abc", "cab").word.sub_map({"a": "1", "b": "2"}).tolist(),
                         ["12c", "c12"])

    def test_split_vi_tone(self):
        self.assertEqual(self.w("việt", "tốt", "hello").word.split_vi_tone.tolist(),
                         ["viê.t", "tô't", "hello"])

    def test_split_vi_diacritical(self):
        self.assertEqual(self.w("đẹp", "ăn", "hello").word.split_vi_diacritical.tolist(),
                         ["d-ẹp", "a{n", "hello"])

    def test_move_vi_tone_to_last(self):
        self.assertEqual(self.w("viê.t", "hello").word.move_vi_tone_to_last.tolist(),
                         ["viêt.", "hello"])

    def test_truncate_first_vi_mark(self):
        self.assertEqual(self.w("viê.t", "vie^t", "d-ep", "hello").word.truncate_first_vi_mark
                         .tolist(), ["viê.", "vie^", "d-", "hello"])

    def test_remove_vietnamese_tone(self):
        self.assertEqual(self.w("việt", "Tốt", "hello").word.remove_vietnamese_tone.tolist(),
                         ["viêt", "Tôt", "hello"])

    def test_extract_vietnamese_tone(self):
        self.assertEqual(self.w("việt", "tốt", "hello").word.extract_vietnamese_tone.tolist(),
                         [".", "'", ""])


if __name__ == "__main__":
    unittest.main()
