"""Functions dealing with dataclass columns."""


import dataclasses
import pandas as pd

from mt import tp


__all__ = ["pack_dataclass", "unpack_dataclass"]


def pack_dataclass(
    df, output_col: str, klass, field_map: tp.Optional[dict] = None, drop: bool = False
):
    """Packs a list of component fields of a dataframe as a dataclass field.

    Parameters
    ----------
    df : pandas.DataFrame
        the input dataframe to work on
    output_col : str
        the name of the output dataclass field
    klass : object
        the dtype of the output dataclass field
    field_map: dict, optional
        if provided, a mapping that maps each component field's name into a field of the dataclass
    drop: bool
        whether or not (default) to drop the component fields in the dataframe

    Returns
    -------
    pandas.DataFrame
        the output dataframe, where the dataclass field is appended to the input dataframe, and
        optionally the component fields dropped
    """

    if len(df) == 0:
        raise ValueError("The dataframe is empty.")

    if not dataclasses.is_dataclass(klass):
        raise TypeError(
            "The provided class is not a dataclass: '{}'.".format(type(klass))
        )

    if field_map is None:  # generate automatically
        fields = dataclasses.fields(klass)
        field_map = {field.name: field.name for field in fields}

    for col in field_map:
        if not col in df.columns:
            raise ValueError(
                "Field '{}' does not exist in the given dataframe with columns '{}'.".format(
                    col, df.columns
                )
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

    Parameters
    ----------
    df : pandas.DataFrame
        the input dataframe to work on
    input_col : str
        the name of the input dataclass field
    klass : object, optional
        the dtype of the input dataclass field. To be detected if not provided.
    field_map: dict, optional
        if provided, a mapping that maps each component field's name into a field of the dataclass
    drop: bool
        whether or not (default) to drop the dataclass field after unpacking

    Returns
    -------
    pandas.DataFrame
        the output dataframe, where new component fields are generated, and optionally the
        dataclass field dropped
    """

    if len(df) == 0:
        raise ValueError("The dataframe is empty.")

    if input_col not in df.columns:
        raise ValueError(
            "Field '{}' does not exist in the given dataframe with columns '{}'.".format(
                input_col, df.columns
            )
        )

    if klass is None:
        klass = type(df.iloc[0][input_col])
    if not dataclasses.is_dataclass(klass):
        raise TypeError(
            "The provided class is not a dataclass: '{}'.".format(type(klass))
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
