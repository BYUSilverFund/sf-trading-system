from datetime import date

import polars as pl

from silverfund.data_access_layer.barra_returns import load_barra_returns
from silverfund.data_access_layer.universe import load_universe
from silverfund.enums import Interval


def load_benchmark(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    # Load universe
    universe = load_universe(interval=interval, start_date=start_date, end_date=end_date)

    # Load barra returns
    barra = load_barra_returns(interval=interval, start_date=start_date, end_date=end_date)

    # Merge universe and returns
    benchmark = universe.join(barra, on=["date", "barrid"], how="left").sort(["date", "barrid"])

    # Get total market cap by day
    total_market_cap = (
        benchmark.group_by("date").agg(pl.col("mktcap").sum().alias("total_mktcap")).sort("date")
    )

    # Create benchmark weights
    benchmark = (
        benchmark.join(total_market_cap, on="date", how="left")
        .with_columns((pl.col("mktcap") / pl.col("total_mktcap")).alias("weight"))
        .select(["date", "barrid", "weight", "ret"])
    )

    return benchmark
