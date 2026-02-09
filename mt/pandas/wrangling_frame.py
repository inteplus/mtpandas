# pylint: disable=import-outside-toplevel, line-too-long

"""Definition and implementation of a wrangling frame."""

from mt import tp, pd
from mt.base import LogicError


class WranglingFrame(object):
    """Wrangling frame.

    A wrangling frame is an object that wraps a dataframe which is intended to be used for data
    wrangling. This dataframe can be checked out, transformed, and then checked in. MT will update
    the definition later.

    Parameters
    ----------
    df : pandas.DataFrame
        initial dataframe for data wrangling
    """

    def __init__(self, df: pd.DataFrame):
        self.wrangling = False
        self.df = df.copy()
        if "unwrangled_reason" not in self.df.columns:
            self.df["unwrangled_reason"] = None

    def n_rows_to_wrangle(self) -> int:
        """Returns the number of rows left to wrangle."""
        s = self.df["unwrangled_reason"].isnull()
        return s.sum()

    def empty_column(self, col: str) -> bool:
        """Checks if a column is empty or not.

        A column is empty if it either does not exist or has no non-null value.

        Parameters
        ----------
        col : str
            name of the column to check

        Returns
        -------
        bool
            whether or not the column is empty
        """
        if col not in self.df.columns:
            return True

        s = self.df["unwrangled_reason"].isnull()
        return self.df[s][col].isna().all()

    def checkout(self, max_n_rows: tp.Optional[int] = None) -> pd.DataFrame:
        """Checks out unwrangled rows for processing.

        Parameters
        ----------
        max_n_rows : int, optional
            if given, the maximum number of rows to checkout
        """
        if self.wrangling:
            raise LogicError(
                "Wrangling of a managed dataframe cannot be nested.",
                debug={"df": self.df},
            )
        self.wrangling = True

        s = self.df["unwrangled_reason"].isnull()
        self.last_null_count = s.sum()
        if not pd.isna(max_n_rows):
            s &= s.cumsum() <= max_n_rows

        self.s = s
        return self.df[self.s].copy()

    def checkin(self, df: tp.Optional[pd.DataFrame] = None):
        if not self.wrangling:
            raise LogicError(
                "The dataframe has not been checked out yet.",
                debug={"df": self.df},
            )
        if df is not None:
            n_before = len(self.df)
            if self.s.sum() == len(self.df):
                # If all rows are checked out, we can just replace the dataframe.
                self.df = df.copy()
            else:
                self.df = pd.concat([self.df[~self.s], df])
            n_after = len(self.df)
            if n_before != n_after:
                raise LogicError(
                    "The number of rows of the dataframe has changed.",
                    debug={
                        "n_before": n_before,
                        "n_after": n_after,
                        "columns": self.df.columns,
                    },
                )
        del self.s
        self.wrangling = False
