"""Batch processing a dataframe."""

import pandas as pd

from mt import tp
from mt.base.dataframe_processing import (
    default_preprocess,
    default_batchprocess,
    default_postprocess,
    sample_rows,
    process_dataframe_impl,
)


__all__ = [
    "default_preprocess",
    "default_batchprocess",
    "default_postprocess",
    "sample_rows",
    "process_dataframe",
]


async def process_dataframe(
    df: pd.DataFrame,
    preprocess_func,
    batchprocess_func=None,
    postprocess_func=None,
    rng_seed: int = 0,
    num_iters: tp.Optional[int] = None,
    preprocess_args: tuple = (),
    preprocess_kwargs: dict = {},
    batchprocess_args: tuple = (),
    batchprocess_kwargs: dict = {},
    postprocess_args: tuple = (),
    postprocess_kwargs: dict = {},
    skip_null: bool = False,
    iter_policy: str = "sequential",
    resampling_col: tp.Optional[str] = None,
    batch_size: int = 32,
    s3_profile: tp.Optional[str] = None,
    max_concurrency: int = 16,
    context_vars: dict = {},
    logger=None,
):
    """An asyn function that does batch processing a dataframe.

    The functionality provided here addresses the following situation. The user has a dataframe in
    which each row represents an event or an image. There is a need to take some fields of each row
    and to convert them into tensors to feed one or more models for training, prediction, validation
    or evaluation purposes. Upon applying the tensors to a model and getting some results, there is
    a need to transform output tensors back to some fields. In addition, the model(s) can only
    operate in batches, rather than on individual items.

    To address this situation, the user needs to provide 3 asyn functions: `preprocess`,
    `batchprocess` and `postprocess`. `preprocess` applies to each row for converting some fields of
    the row into a dictionary of tensors. Tensors of the same name are stacked up to form a
    dictionary of batch tensors, and then are fed to `batchprocess` for batch processing using the
    model(s). The output batch tensors are unstacked into individual tensors and then fed to
    `postprocess` to convert them back into pandas.Series representing fields for each row. Finally,
    these new fields are concatenated to form an output dataframe.

    The three above functions have a default implementation, namely :func:`default_preprocess`,
    :func:`default_batchprocess` and :func:`default_postprocess`, respectively. The user must make
    sure the APIs of their functions match with those of the default ones. Note that it is possible
    that during preprocessing or batchprocessing a row the function can skip batch-processing and
    postprocessing altogether and returns an output series corresponding to each input row.

    Internally, we use the BeeHive concurrency model to address the problem. The queen bee is
    responsible for forming batches, batch processing, and forming the output dataframe, the
    worker bees that she spawns are responsible for doing preprocessing and postprocessing works
    of each row.

    Parameters
    ----------
    df : pandas.DataFrame
        an input unindexed dataframe
    preprocess_func : function
        the preprocessing function
    batchprocess_func : function, optional
        the function for batch-processing. If not provided, the preprocess function must returned
        postprocessed pandas.Series instances.
    postprocess_func : function, optional
        the postrocessing function. If not provided, the preprocess and batchprocess functions
        must make sure that every row is processed and a pandas.Series is returned.
    rng_seed : int, optional
        seed for making RNGs
    num_iters : int, optional
        number of iterations or equivalently number of rows selected during the call. If not
        provided, it is set to the number of rows of `df`.
    preprocess_args : tuple, optional
        positional arguments to be passed as-is to :func:`preprocess`
    preprocess_kwargs : dict, optional
        keyword arguments to be passed as-is to :func:`preprocess`
    batchprocess_args : tuple, optional
        positional arguments to be passed as-is to :func:`batchprocess`
    batchprocess_kwargs : dict, optional
        keyword arguments to be passed as-is to :func:`batchprocess`
    postprocess_args : tuple, optional
        positional arguments to be passed as-is to :func:`postprocess`
    postprocess_kwargs : dict, optional
        keyword arguments to be passed as-is to :func:`postprocess`
    skip_null : bool
        If True, any None returned value from the provided functions will be considered as a
        trigger to skip the row. Otherwise, an exception is raised as usual.
    iter_policy : {'sequential', 'resampling'}
        policy for iterating the rows. If 'sequential' is given, the items are iterated
        sequentially from the first row to the last row and then back to the first row if required.
        If 'resampling' is given, then the dataframe column provided by the `resampling_col`
        argument provides the resampling weights. Rows are resampled randomly using these weights
        as the resampling distribution.
    resampling_col : str
        name of the column/field containing the resampling weights. Only valid if `iter_policy` is
        'resampling'.
    batch_size : int
        maximum batch size for each batch that is formed internally
    s3_profile : str, optional
        The AWS S3 profile to be used so that we can spawn an S3 client for each newly created
        subprocess. If not provided, the default profile will be used.
    max_concurrency : int
        the maximum number of concurrent tasks each worker bee handles at a time
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
        Variable 's3_client' must exist and hold an enter-result of an async with statement
        invoking :func:`mt.base.s3.create_s3_client`. In asynchronous mode, variable
        'http_session' must exist and hold an enter-result of an async with statement invoking
        :func:`mt.base.http.create_http_session`.
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes

    Returns
    -------
    df : pandas.DataFrame
        an output unindexed dataframe

    Notes
    -----
    The function only works in asynchronous mode. That means `context_vars['async'] is True` is
    required.
    """

    return await process_dataframe_impl(
        df,
        preprocess_func,
        batchprocess_func=batchprocess_func,
        postprocess_func=postprocess_func,
        rng_seed=rng_seed,
        num_iters=num_iters,
        preprocess_args=preprocess_args,
        preprocess_kwargs=preprocess_kwargs,
        batchprocess_args=batchprocess_args,
        batchprocess_kwargs=batchprocess_kwargs,
        postprocess_args=postprocess_args,
        postprocess_kwargs=postprocess_kwargs,
        skip_null=skip_null,
        iter_policy=iter_policy,
        resampling_col=resampling_col,
        batch_size=batch_size,
        s3_profile=s3_profile,
        max_concurrency=max_concurrency,
        context_vars=context_vars,
        logger=logger,
    )
