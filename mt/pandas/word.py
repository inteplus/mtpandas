"""Custom word accessor for pandas.

Importing :mod:`mt.pandas` registers the accessor as ``Series.word``. It provides helpers for
series of single words (English or Vietnamese), such as letter n-grams and Vietnamese tone and
diacritical mark handling.
"""

import pandas.api.extensions as _pae
import re as _re


__all__ = ["WordAccessor"]


def _construct_vi_vowel_tone_inverse_map(vi_vowel_tones, vi_vowel_tone_map):
    res = {}
    for key in vi_vowel_tones:
        for i, ch in enumerate(vi_vowel_tones[key]):
            res[ch] = key + vi_vowel_tone_map[i]
    return res


@_pae.register_series_accessor("word")
class WordAccessor:
    """Accessor for word fields, available as ``Series.word``.

    Each item of the series is expected to be a string representing one word. Vietnamese tones are
    written as the symbols in :attr:`vi_vowel_tone_map` and Vietnamese diacritical letters as the
    two-character codes in :attr:`vi_extended_letter_map`.

    Examples
    --------
    >>> import pandas as pd
    >>> import mt.pandas
    >>> w = pd.Series(["hello", "việt", "a1"])
    >>> w.word.english.tolist()
    [True, False, False]
    >>> w.word.vietnamese.tolist()
    [True, True, False]
    >>> w.word.bigram.tolist()
    [['he', 'el', 'll', 'lo'], ['vi', 'iệ', 'ệt'], ['a1']]
    """

    # ----- data -----

    vi_extended_letters = {"lower": "ăâđêôơư", "upper": "ĂÂĐÊÔƠƯ"}

    vi_extended_letter_map = {
        "ă": "a{",
        "â": "a}",
        "đ": "d-",
        "ê": "e^",
        "ô": "o]",
        "ơ": "o[",
        "ư": "u+",
    }

    vi_vowel_tones = {
        "a": "àáảãạ",
        "ă": "ằắẳẵặ",
        "â": "ầấẩẫậ",
        "e": "èéẻẽẹ",
        "ê": "ềếểễệ",
        "i": "ìíỉĩị",
        "o": "òóỏõọ",
        "ô": "ồốổỗộ",
        "ơ": "ờớởỡợ",
        "u": "ùúủũụ",
        "ư": "ừứửữự",
        "y": "ỳýỷỹỵ",
        "A": "ÀÁẢÃẠ",
        "Ă": "ẰẮẲẴẶ",
        "Â": "ẦẤẨẪẬ",
        "E": "ÈÉẺẼẸ",
        "Ê": "ỀẾỂỄỆ",
        "I": "ÌÍỈĨỊ",
        "O": "ÒÓỎÕỌ",
        "Ô": "ỒỐỔỖỘ",
        "Ơ": "ỜỚỞỠỢ",
        "U": "ÙÚỦŨỤ",
        "Ư": "ỪỨỬỮỰ",
        "Y": "ỲÝỶỸỴ",
    }

    vi_vowel_tone_map = "`'?~."

    vi_vowel_tone_inverse_map = _construct_vi_vowel_tone_inverse_map(
        vi_vowel_tones, vi_vowel_tone_map
    )

    # ----- constructor -----

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    # ----- useful methods -----

    def sub_map(self, substr_map):
        """Substitutes substrings using a dictionary/map.

        For each substring of a word, the substring is replaced with a replacement string. The keys
        of the map are joined with ``|`` into one regular expression, so they must be valid regular
        expression fragments.

        Parameters
        ----------
        substr_map : dict
            a map that maps each substring into a replacement string

        Returns
        -------
        pandas.Series
            the series of words after substitution
        """
        pat = _re.compile("|".join(substr_map.keys()))
        repl = lambda m: substr_map[m.group(0)]
        return self._obj.str.replace(pat, repl, regex=True)

    # ----- properties -----

    @property
    def english(self):
        """Returns a boolean series telling which item is like an English word.

        An item matches if it is made of ASCII letters only (``[a-zA-Z]+``).
        """
        return self._obj.str.match("[a-zA-Z]+$")

    @property
    def vietnamese(self):
        """Returns a boolean series telling which item is like a Vietnamese word.

        An item matches if it is made only of ASCII letters, the Vietnamese extended letters and
        the toned Vietnamese vowels.
        """
        pat = (
            "[a-zA-Z"
            + "".join(self.vi_extended_letters.values())
            + "".join(self.vi_vowel_tones.values())
            + "]+$"
        )
        return self._obj.str.match(pat)

    @property
    def split_vi_tone(self):
        """Splits any Vietnamese toned letter into its base letter followed by a symbol representing
        the tone mark `(`'?~.)`, in each word.

        For example, "ệ" becomes "ê." (base letter "ê" followed by the dot mark) and "á" becomes
        "a'". See also :attr:`split_vi_diacritical`.
        """
        return self.sub_map(self.vi_vowel_tone_inverse_map)

    @property
    def split_vi_diacritical(self):
        """Splits any untoned diacritical Vietnamese letter into its base letter followed by a
        symbol representing the diacritical mark, in each word.

        For example, "ă" becomes "a{" and "đ" becomes "d-". See :attr:`vi_extended_letter_map`.
        """
        return self.sub_map(self.vi_extended_letter_map)

    @property
    def move_vi_tone_to_last(self):
        """Moves the first tone mark to the end of a word.

        It is meant to be applied on words processed by :attr:`split_vi_tone`.
        """
        return self._obj.str.replace(
            r"([^`'\?~\.]*)([`'\?~\.])(.*)",
            lambda x: x.group(1) + x.group(3) + x.group(2),
            regex=True,
        )

    @property
    def truncate_first_vi_mark(self):
        """Truncates each word to the first occurence of a split Vietnamese mark.

        The mark itself is kept, everything after it is dropped.
        """
        return self._obj.str.replace(
            r"([^\{\}\-\^\[\]\+`'\?~\.]*)([\{\}\-\^\[\]\+`'\?~\.])(.*)",
            lambda x: x.group(1) + x.group(2),
            regex=True,
        )

    @property
    def remove_vietnamese_tone(self):
        """Removes the tone marks in each Vietnamese word, keeping the base vowels.

        For example, "ệ" becomes "ê".
        """
        rep = {c: k for k, v in self.vi_vowel_tones.items() for c in v}
        return self.sub_map(rep)

    @property
    def extract_vietnamese_tone(self):
        """Extracts the tone marks `{"`"', "'", "?", "~", "."}` each Vietnamese word.

        Every character that is not a toned vowel is dropped, so the result contains only the tone
        mark symbols found in the word, in order.
        """

        l = "`'?~."
        rep = {v[i]: l[i] for k, v in self.vi_vowel_tones.items() for i in range(5)}
        return self._obj.str.replace(".", lambda m: rep.get(m.group(0), ""), regex=True)

    def ngram(self, n):
        """Returns a list letter n-grams for each word.

        Parameters
        ----------
        n : int
            number n specifying the letter n-gram. Must be integer greater than 1.

        Returns
        -------
        pandas.Series
            each element of the returning series is a list of n-grams of the corresponding element
            in the input series

        Raises
        ------
        ValueError
            if an argument is wrong

        Notes
        -----
        You can use pandas' explode() function to process further. A word shorter than `n` gives
        an empty list.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series(["abcd", "ab"]).word.ngram(3).tolist()
        [['abc', 'bcd'], []]
        """
        if not isinstance(n, int) or n < 2:
            raise ValueError(
                f"Expected n to be integer greater than 1, given {n}."
            )
        return self._obj.apply(lambda x: [x[i : i + n] for i in range(len(x) - n + 1)])

    @property
    def letter(self):
        """Returns a series of lists of letters, one list for each word.

        Examples
        --------
        >>> import pandas as pd
        >>> import mt.pandas
        >>> pd.Series(["việt"]).word.letter.tolist()
        [['v', 'i', 'ệ', 't']]
        """
        return self._obj.apply(lambda x: list(x))

    @property
    def bigram(self):
        """Returns a list of letter bigrams for each word. See :meth:`ngram`."""
        return self.ngram(2)

    @property
    def trigram(self):
        """Returns a list of letter trigrams for each word. See :meth:`ngram`."""
        return self.ngram(3)
