from typing import Protocol

import polars as pl

from silverfund.records import Signal


class SignalConstructor(Protocol):
    """Protocol for functions that construct a Signal from a Polars DataFrame."""

    def __call__(self, data: pl.DataFrame) -> Signal: ...


def momentum(data: pl.DataFrame) -> Signal:
    """Computes the momentum signal based on log returns.

    The momentum signal is calculated by first computing the log returns (`logret`)
    of the `ret` column, then calculating the 11-period rolling sum of the log returns
    within each `barrid` group. The resulting momentum is shifted by one period to ensure
    it represents the previous period's momentum for each asset.

    Args:
        data (pl.DataFrame): A Polars DataFrame containing asset return data,
                              where 'ret' represents the returns of assets.

    Returns:
        Signal: A Signal object containing the 'date', 'barrid', and computed 'mom' (momentum) columns.
    """
    signals = (
        data.with_columns(pl.col("ret").log1p().alias("logret"))
        .with_columns(pl.col("logret").rolling_sum(11, min_periods=11).over("barrid").alias("mom"))
        .with_columns(pl.col("mom").shift(1).over("barrid"))
        .select(["date", "barrid", "mom"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "mom")


def barraBAB(data: pl.DataFrame) -> Signal:
    """
    Computes the 'barraBAB' signal by:
    1. Multiplying predbeta by -1 to get the desired signal (Long low beta and short high beta).

    Args:
        data (pl.DataFrame): A Polars DataFrame containing asset return data,
                              where 'ret' represents the returns of assets.
                              
    Returns:
        Signal: A Signal object containing the 'date', 'barrid', and changed 'predbeta' columns.
    """

    signals = (
        data.with_columns(pl.col("predbeta") * -1)
        .select(["date", "barrid", "predbeta"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "predbeta")

def barraBAB_noskew(data: pl.DataFrame) -> Signal:
    """
    Computes the 'barraBAB' signal by:
    1. Ranking 'predbeta' within each date and scaling to [0, 1].
    2. Applying the inverse CDF of the standard normal distribution
       to remove skewness.
    3. Multiplying by -1 to get the desired signal (Long low beta and short high beta).

    Args:
        data (pl.DataFrame): A Polars DataFrame containing asset return data,
                              where 'ret' represents the returns of assets.

    Returns:
        Signal: A Signal object containing the 'date', 'barrid', and computed 'noskew_predbeta' columns.
    """
    from scipy.stats import norm
    import numpy as np

    ranked_df = (
        data.sort(["date", "predbeta"])
        .with_columns(
            (
                (pl.col("predbeta").rank("average").over("date") - 1)
                / (pl.col("predbeta").count().over("date") - 1)
            ).alias("rank_scaled")
        )
    )

    transformed_df = ranked_df.with_columns(
        pl.col("rank_scaled").map_elements(lambda x: norm.ppf(np.clip(x, 1e-6, 1 - 1e-6)), return_dtype=pl.Float64).alias("noskew_predbeta")
    )

    signals = transformed_df.with_columns(pl.col("noskew_predbeta") * -1).select(["date", "barrid", "noskew_predbeta"]).sort(["barrid", "date"])

    return Signal(signals, "noskew_predbeta")


def lowBeta(data: pl.DataFrame) -> Signal:
    signals = (
        data.with_columns(
            pl.col("predbeta")
            .qcut(10, labels=[str(x) for x in range(10)])
            .over("date")
            .alias("bin")
        )
        .with_columns(
            pl.when(pl.col("bin") == "0") 
            .then(pl.col("predbeta") * 1) 
            .otherwise(pl.lit(None))
            .alias("predbeta")
        )
        .select(["date", "barrid", "predbeta"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "predbeta")

def noSkewLowBeta(data: pl.DataFrame) -> Signal:
    """
    Computes a low-beta signal with skewness removed:
    1. Ranks 'predbeta' within each date and scales to [0, 1].
    2. Applies the inverse CDF of the standard normal distribution.
    3. Selects only the lowest decile (bin "0").
    
    Args:
        data (pl.DataFrame): A Polars DataFrame containing 'predbeta' values.

    Returns:
        Signal: A Signal object containing 'date', 'barrid', and computed 'noskew_predbeta' columns.
    """
    from scipy.stats import norm
    import numpy as np

    ranked_df = (
        data.sort(["date", "predbeta"])
        .with_columns(
            (
                (pl.col("predbeta").rank("average").over("date") - 1)
                / (pl.col("predbeta").count().over("date") - 1)
            ).alias("rank_scaled")
        )
    )

    transformed_df = ranked_df.with_columns(
        pl.col("rank_scaled").map_elements(lambda x: norm.ppf(np.clip(x, 1e-6, 1 - 1e-6)), return_dtype=pl.Float64).alias("noskew_predbeta")
    )

    # Assign deciles based on 'predbeta' ranking
    binned_df = transformed_df.with_columns(
        pl.col("rank_scaled")
        .qcut(10, labels=[str(x) for x in range(10)])
        .over("date")
        .alias("bin")
    )

    # Keep only the lowest beta decile (bin "0")
    filtered_df = (
        binned_df
        .with_columns(
            pl.when(pl.col("bin") == "0")
            .then(pl.col("noskew_predbeta"))
            .otherwise(pl.lit(None))
            .alias("noskew_predbeta")
        )
        .select(["date", "barrid", "noskew_predbeta"])
        .sort(["barrid", "date"])
    )

    return Signal(filtered_df, "noskew_predbeta")
