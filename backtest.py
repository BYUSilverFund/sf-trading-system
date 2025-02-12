from datetime import date
from functools import partial

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import *
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)

strategy = Strategy(
    signal_constructor=momentum,
    score_constructor=partial(z_score, col="mom", over="barrid"),
    alpha_constructor=grindold_kahn,
    portfolio_constructor=mean_variance_efficient,
    constraints=[full_investment, no_buying_on_margin, long_only, unit_beta],
)

# Instantiate backtester
bt = Backtester(start_date=start_date, end_date=end_date, strategy=strategy)

# Run sequentially
pnl = bt.run_sequential()
print("-" * 20 + " Asset Returns " + "-" * 20)
print(pnl)

# -------------------- Plot --------------------
daily_returns = (
    pnl.with_columns((pl.col("weight") * pl.col("fwd_ret")).alias("contribution"))
    .group_by("date")
    .agg(pl.col("contribution").sum().alias("portfolio_ret"))
    .with_columns((pl.col("portfolio_ret") / 100).log1p().cum_sum().alias("portfolio_cumret"))
    .with_columns(pl.col("portfolio_cumret") * 100)  # put into percent space
    .sort("date")
)

# Table
print("-" * 20 + " Portfolio Returns " + "-" * 20)
print(daily_returns)

# Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=daily_returns, x="date", y="portfolio_cumret")
plt.title("Monthly Momentum Backtest")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Returns (%)")
plt.savefig("backtest.png", dpi=300)
