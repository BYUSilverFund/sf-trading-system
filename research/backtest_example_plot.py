import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Load data
pnl = pl.read_parquet("research/backtest_example.parquet")

# Aggregate to portfolio level
daily_returns = (
    pnl.with_columns((pl.col("weight") * pl.col("fwd_ret")).alias("contribution"))
    .group_by("date")
    .agg(pl.col("contribution").sum().alias("portfolio_ret"))
    .sort("date")
    .with_columns((pl.col("portfolio_ret") / 100).log1p().cum_sum().alias("portfolio_cumret"))
    .with_columns(pl.col("portfolio_cumret") * 100)  # put into percent space
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
plt.savefig("research/backtest_example.png", dpi=300)
