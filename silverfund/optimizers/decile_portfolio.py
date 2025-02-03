import numpy as np
import polars as pl

from silverfund.enums import Weighting


def decile_portfolio(chunk: pl.DataFrame, signal: str, weighting: Weighting):

    # Bin portfolios into 10 equal sized deciles.
    labels = [str(x) for x in range(10)]

    chunk = chunk.with_columns(
        pl.col(signal).qcut(10, labels=labels).cast(pl.Int32).over("date").alias("bin")
    )

    # Seperate portfolios
    portfolios = [
        chunk.filter(pl.col("bin") == i).select(["date", "permno", signal, "bin"])
        for i in range(10)
    ]

    # Weights
    if weighting == Weighting.EQUAL:
        portfolios = [
            portfolio.with_columns(pl.lit(1 / len(portfolio)).alias("weight"))
            for portfolio in portfolios
        ]

    return portfolios
