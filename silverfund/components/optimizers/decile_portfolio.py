import numpy as np
import polars as pl

from silverfund.components.enums import Weighting


def decile_portfolio(chunk: pl.DataFrame, signal: str, weighting: Weighting):
    chunk = chunk.drop_nulls()

    # Calculate decile percentiles
    percentiles = np.linspace(0.1, 1, 10)
    values = [chunk[signal].quantile(percentile) for percentile in percentiles]

    # Dynamically build the conditions for decile bins
    bins_expr = pl.when(pl.col(signal) <= values[0]).then(0)
    for i in range(1, len(values)):
        bins_expr = bins_expr.when(pl.col(signal) <= values[i]).then(i)

    # Add the 'bins' column to the DataFrame
    binned_chunk = chunk.with_columns(bins_expr.alias("bin"))

    # Seperate portfolios
    portfolios = [
        binned_chunk.filter(pl.col("bin") == i).select(["date", "permno", signal, "bin"])
        for i in range(10)
    ]

    # Weights
    if weighting == Weighting.EQUAL:
        portfolios = [
            portfolio.with_columns(pl.lit(1 / len(portfolio)).alias("weight"))
            for portfolio in portfolios
        ]

    return portfolios
