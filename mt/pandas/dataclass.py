"""Functions dealing with dataclass columns.

A dataclass column is a column whose items are instances of the same :mod:`dataclasses` class.
:func:`pack_dataclass` builds such a column from several plain columns and :func:`unpack_dataclass`
does the reverse.
"""


import dataclasses
import pandas as pd

from mt import tp


__all__ = ["pack_dataclass", "unpack_dataclass"]


def pack_dataclass(
    df, output_col: str, klass, field_map: tp.Optional[dict] = None, drop: bool = False
):
    """Packs a list of component fields of a dataframe as a dataclass field.

    For each row, an instance of `klass` is built by passing the values of the component columns
    as keyword arguments, and stored in column `output_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        the input dataframe to work on. Note that column `output_col` is added to `df` in place,
        even when `drop` is True.
    output_col : str
        the name of the output dataclass field
    klass : type
        the dataclass type of the output dataclass field
    field_map: dict, optional
        if provided, a mapping that maps each component column's name into a field name of the
        dataclass. If not provided, every field of the dataclass is taken from the column of the
        same name.
    drop: bool
        whether or not to drop the component columns in the output dataframe. Default is False.

    Returns
    -------
    pandas.DataFrame
        the output dataframe, where the dataclass field is appended to the input dataframe, and
        optionally the component fields dropped. If `drop` is False it is `df` itself.

    Raises
    ------
    ValueError
        if the dataframe is empty or a component column does not exist
    TypeError
        if `klass` is not a dataclass

    See Also
    --------
    unpack_dataclass
        the reverse function

    Examples
    --------
    >>> import dataclasses
    >>> import pandas as pd
    >>> from mt.pandas import pack_dataclass
    >>> @dataclasses.dataclass
    ... class Point:
    ...     x: int
    ...     y: int
    >>> df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    >>> pack_dataclass(df, "p", Point, drop=True)
       z            p
    0  5  Point(x=1, y=3)
    1  6  Point(x=2, y=4)
    """

    if len(df) == 0:
        raise ValueError("The dataframe is empty.")

    if not dataclasses.is_dataclass(klass):
        raise TypeError(
            f"The provided class is not a dataclass: '{type(klass)}'."
        )

    if field_map is None:  # generate automatically
        fields = dataclasses.fields(klass)
        field_map = {field.name: field.name for field in fields}

    for col in field_map:
        if not col in df.columns:
            raise ValueError(
                f"Field '{col}' does not exist in the given dataframe with columns '{df.columns}'."
            )

    def func(row):
        return klass(**{field: row[col] for col, field in field_map.items()})

    df2 = df
    df2[output_col] = df2.apply(func, axis=1)
    if drop:
        df2 = df2.drop(list(field_map.keys()), axis=1)

    return df2


def unpack_dataclass(
    df,
    input_col: str,
    klass: tp.Optional[object] = None,
    field_map: tp.Optional[dict] = None,
    drop: bool = False,
):
    """Unpacks a dataclass field of a dataframe as a list of component fields.

    For each dataclass field, a new column with the same name is created from the attribute of
    each item.

    Parameters
    ----------
    df : pandas.DataFrame
        the input dataframe to work on. Note that the component columns are added to `df` in
        place, even when `drop` is True.
    input_col : str
        the name of the input dataclass field
    klass : type, optional
        the dataclass type of the input dataclass field. It is detected from the first row if not
        provided.
    field_map: dict, optional
        if provided, a mapping that maps each output column's name into a field name of the
        dataclass. If not provided, every field of the dataclass becomes a column of the same
        name.
    drop: bool
        whether or not to drop the dataclass field in the output dataframe. Default is False.

    Returns
    -------
    pandas.DataFrame
        the output dataframe, where new component fields are generated, and optionally the
        dataclass field dropped. If `drop` is False it is `df` itself.

    Raises
    ------
    ValueError
        if the dataframe is empty or `input_col` does not exist
    TypeError
        if `klass` is not a dataclass

    See Also
    --------
    pack_dataclass
        the reverse function

    Examples
    --------
    >>> import dataclasses
    >>> import pandas as pd
    >>> from mt.pandas import unpack_dataclass
    >>> @dataclasses.dataclass
    ... class Point:
    ...     x: int
    ...     y: int
    >>> df = pd.DataFrame({"p": [Point(1, 3), Point(2, 4)]})
    >>> unpack_dataclass(df, "p", drop=True)
       x  y
    0  1  3
    1  2  4
    """

    if len(df) == 0:
        raise ValueError("The dataframe is empty.")

    if input_col not in df.columns:
        raise ValueError(
            f"Field '{input_col}' does not exist in the given dataframe with columns "
            f"'{df.columns}'."
        )

    if klass is None:
        klass = type(df.iloc[0][input_col])
    if not dataclasses.is_dataclass(klass):
        raise TypeError(
            f"The provided class is not a dataclass: '{type(klass)}'."
        )

    if field_map is None:  # generate automatically
        fields = dataclasses.fields(klass)
        field_map = {field.name: field.name for field in fields}

    df2 = df
    for col, field in field_map.items():
        df2[col] = df2[input_col].apply(lambda x: getattr(x, field))
    if drop:
        df2 = df2.drop(input_col, axis=1)

    return df2
