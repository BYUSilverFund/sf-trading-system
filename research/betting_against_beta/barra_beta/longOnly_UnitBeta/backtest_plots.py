import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Load data
pnl = pl.read_parquet("barraBAB_backtest.parquet")

# Aggregate to portfolio level
portfolio_returns = (
    pnl.with_columns((pl.col("weight") * pl.col("fwd_ret").fill_null(0)).alias("contribution"))
    .group_by("date")
    .agg(
        pl.col("contribution").sum().alias("portfolio_ret"),
        pl.col("barrid").count().alias("n_assets"),
    )
    .sort("date")
    .with_columns((pl.col("portfolio_ret") / 100).cum_sum().alias("portfolio_cumret"))
    .with_columns(pl.col("portfolio_cumret") * 100)  # put into percent space
)

# Sharpe
portfolio_ret = portfolio_returns["portfolio_ret"].mean()
portfolio_vol = portfolio_returns["portfolio_ret"].std()
portfolio_sharpe = (portfolio_ret / portfolio_vol) * np.sqrt(12)
print(f"Sharpe: {portfolio_sharpe:.4f}")

# Table
print("-" * 20 + " Portfolio Returns " + "-" * 20)
print(portfolio_returns)

# Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=portfolio_returns, x="date", y="portfolio_cumret")
plt.title("Monthly BAB Backtest")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Returns (%)")
plt.savefig("backtest_results.png", dpi=300)