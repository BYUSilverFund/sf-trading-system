from datetime import date

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.constraints import full_investment, long_only, no_buying_on_margin
from silverfund.enums import Interval
from silverfund.portfolios import mve_parallel
from silverfund.records import Alpha

start_date = date(2023, 12, 15)
end_date = date(2023, 12, 31)

universe = dal.load_universe(interval=Interval.DAILY, start_date=start_date, end_date=end_date)

data = dal.load_barra_returns(interval=Interval.DAILY, start_date=start_date, end_date=end_date)

alphas = (
    universe
    # Join on universe
    .join(data, how="left", on=["date", "barrid"])
    .with_columns(
        pl.col("ret").log1p().rolling_sum(window_size=230).over("barrid").alias("momentum")
    )
    .with_columns(pl.col("momentum").over("barrid").shift(22))
    .fill_null(0)
    .sort(["barrid", "date"])
    .select(["date", "barrid", pl.col("momentum").alias("alpha")])
)

weights = mve_parallel(
    start_date=start_date,
    end_date=end_date,
    alphas=Alpha(alphas),
    constraints=[full_investment, long_only, no_buying_on_margin],
)

results = (
    alphas
    # Join MVE weights
    .join(weights, on=["date", "barrid"], how="left")
)

print(results)
