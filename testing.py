from datetime import date

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.alphas import Alpha, grindold_kahn
from silverfund.constraints import *
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import momentum

start_date = date(2023, 1, 1)
end_date = date(2024, 12, 31)

data = dal.load_barra_returns(interval=Interval.MONTHLY, start_date=start_date, end_date=end_date)

universe = dal.load_monthly_universe(start_date=start_date, end_date=end_date)

data = universe.join(data, on=["date", "barrid"], how="left").sort(["barrid", "date"])

print(data.filter(pl.col("date") == pl.col("date").last()).sort("barrid"))

signals = momentum(data)
print(signals.filter(pl.col("date") == pl.col("date").last()).sort("barrid"))

scores = z_score(signals, "mom")
print(scores.filter(pl.col("date") == pl.col("date").last()).sort("barrid"))

alphas = grindold_kahn(scores)
print(alphas.filter(pl.col("date") == pl.col("date").last()).sort("barrid"))

periods = universe["date"].unique().sort().to_list()
portfolios = []
for period in [periods[-1]]:
    print("PERIOD", period)
    # Get portfolio constructor parameters
    period_barrids = universe.filter(pl.col("date") == period)["barrid"].sort().to_list()
    period_alphas = Alpha(alphas.filter(pl.col("date") == period).sort(["barrid"]))

    # Construct period portfolio
    portfolio = mean_variance_efficient(
        period=period,
        barrids=period_barrids,
        alphas=period_alphas,
        constraints=[full_investment],
    )
    print(portfolio)
    break
