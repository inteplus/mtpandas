# pylint: disable=import-outside-toplevel, line-too-long

"""Definition and implementation of a wrangling frame.

See :class:`WranglingFrame` for the checkout/checkin workflow.
"""

from mt import tp, pd, logg
from mt.base import LogicError


class WranglingFrame(object):
    """Wrangling frame.

    A wrangling frame is an object that wraps a dataframe which is intended to be used for data
    wrangling. This dataframe can be checked out, transformed, and then checked in. MT will update
    the definition later.

    The wrapped dataframe (attribute `df`) is a copy of the input one with an extra column
    ``unwrangled_reason`` (added if missing). A row whose ``unwrangled_reason`` is null still
    needs to be wrangled. :meth:`checkout` returns those rows, the caller transforms them and
    marks the rows that cannot be wrangled by writing a reason in ``unwrangled_reason``, and
    :meth:`checkin` puts the rows back. Rows with a non-null reason are not checked out again.

    Parameters
    ----------
    df : pandas.DataFrame
        initial dataframe for data wrangling. It is copied.

    Attributes
    ----------
    wrangling : bool
        whether or not a checkout is in progress
    df : pandas.DataFrame
        the wrapped dataframe

    Examples
    --------
    >>> import pandas as pd
    >>> from mt.pandas import WranglingFrame
    >>> wf = WranglingFrame(pd.DataFrame({"a": [1, 2, 3]}))
    >>> wf.n_rows_to_wrangle()
    3
    >>> df = wf.checkout(max_n_rows=2)
    >>> df
       a unwrangled_reason
    0  1              None
    1  2              None
    >>> df["unwrangled_reason"] = ["too small", None]
    >>> df["b"] = [None, 5]
    >>> wf.checkin(df)
    >>> wf.df
       a unwrangled_reason    b
    2  3              None  NaN
    0  1        too small  NaN
    1  2              None  5.0
    >>> wf.n_rows_to_wrangle()
    2
    """

    def __init__(self, df: pd.DataFrame):
        self.wrangling = False
        self.df = df.copy()
        if "unwrangled_reason" not in self.df.columns:
            self.df["unwrangled_reason"] = None

    def n_rows_to_wrangle(self) -> int:
        """Returns the number of rows left to wrangle, i.e. with a null ``unwrangled_reason``."""
        s = self.df["unwrangled_reason"].isnull()
        return s.sum()

    def empty_column(self, col: str) -> bool:
        """Checks if a column is empty or not.

        A column is empty if it either does not exist or has no non-null value. Only the rows
        left to wrangle (null ``unwrangled_reason``) are considered.

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

        Only rows whose ``unwrangled_reason`` is null are returned. The frame stays in the
        wrangling state until :meth:`checkin` is called.

        Parameters
        ----------
        max_n_rows : int, optional
            if given, the maximum number of rows to checkout. The first rows are taken.

        Returns
        -------
        pandas.DataFrame
            a copy of the rows to wrangle, with the same index as in the wrapped dataframe

        Raises
        ------
        mt.base.LogicError
            if a previous checkout has not been checked in yet (checkouts cannot be nested)
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
        """Checks in the rows that have been checked out.

        The checked-in rows replace the checked-out ones. Columns that are new to the wrapped
        dataframe are added (missing values are null). Rows are re-ordered: the rows that were not
        checked out come first, followed by the checked-in rows.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            the transformed dataframe returned by :meth:`checkout`. It must have the same number
            of rows as the checked-out one. If not provided, the wrapped dataframe is left
            unchanged and the wrangling state is just ended.

        Raises
        ------
        mt.base.LogicError
            if there is no checkout in progress, or if the number of rows of the wrapped dataframe
            changes after checking in
        """
        if not self.wrangling:
            raise LogicError(
                "The dataframe has not been checked out yet.",
                debug={"df": self.df},
            )
        if df is not None:
            for x in df.columns:
                if x not in self.df.columns:
                    self.df[x] = None
            n_before = len(self.df)
            df2 = self.df[~self.s]
            if len(df2) == 0:
                self.df = df.copy()
            else:
                self.df = pd.concat([df2, df])
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

    def summarise(self, logger: tp.Optional[logg.IndentedLoggerAdapter] = None):
        """Summarises the wrangling frame.

        Logs the number of records, the columns and the count of each ``unwrangled_reason``.

        Parameters
        ----------
        logger : mt.logg.IndentedLoggerAdapter, optional
            logger to write to. If not provided, nothing is done.
        """
        if not logger:
            return
        with logger.scoped_info(f"Wrangling frame of {len(self.df)} records"):
            logger.info(f"Columns: {self.df.columns}")
            if "unwrangled_reason" in self.df.columns:
                df = self.df.groupby("unwrangled_reason", dropna=False).size()
                logger.info(f"Reasons: {df}")
