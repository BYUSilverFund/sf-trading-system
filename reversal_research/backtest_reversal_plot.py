import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Load data

# pnl = pl.read_parquet(
#     "/home/bwaits/Research/sf-trading-system/reversal_research/classic_reversal_full_sample.parquet"
# )

pnl = pl.read_parquet(
    "/home/bwaits/Research/sf-trading-system/reversal_research/barra_reversal_full_sample2.parquet"
)

# pnl = pl.read_parquet(
#     "/home/bwaits/Research/sf-trading-system/reversal_research/reversal_1yr_neg_test.parquet"
# )

# Aggregate to portfolio level
portfolio_returns = (
    pnl.with_columns((pl.col("weight") * pl.col("fwd_ret")).alias("contribution"))
    .group_by("date")
    .agg(pl.col("contribution").sum().alias("portfolio_ret"))
    .sort("date")
    .with_columns((pl.col("portfolio_ret") / 100).log1p().cum_sum().alias("portfolio_cumret"))
    .with_columns(pl.col("portfolio_cumret") * 100)  # put into percent space
)

sharpe = (
    portfolio_returns["portfolio_ret"].mean() / portfolio_returns["portfolio_ret"].std()
) * np.sqrt(12)
print("Backtested Sharpe Ratio: ", sharpe)


# Table
print("-" * 20 + " Portfolio Returns " + "-" * 20)
print(portfolio_returns)

# Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=portfolio_returns, x="date", y="portfolio_cumret")
plt.title(f"Barra Residual Reversal Backtest, Sharpe Ratio: {sharpe:.2f}")
# plt.title(f"Classic Reversal Backtest, Sharpe Ratio: {sharpe:.2f}")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Returns (%)")
# plt.savefig("classic_reversal_full_sample.png", dpi=300)
plt.savefig("barra_reversal_full_sample2.png", dpi=300)
# plt.savefig("barra_reversal_1yr.png", dpi=300)
# plt.savefig("reversal_1yr_neg_test.png", dpi=300)
