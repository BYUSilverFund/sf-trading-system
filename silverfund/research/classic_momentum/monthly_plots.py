from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

results_folder = Path(
    "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/classic_momentum/results"
)

pnl = pl.read_parquet(results_folder / "monthly_momentum_bt.parquet")

# Calculate sharpe ratio
portfolio_er = pnl.select("portfolio_ret").mean()
portfolio_vol = pnl.select("portfolio_ret").std()

annual_factor = 12 / np.sqrt(12)

sharpe = (portfolio_er / portfolio_vol) * annual_factor

print(f"Annualized Sharpe:", round(sharpe.item(), 4))

# Cummulative Sum Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=pnl, x="date", y="cumsum")
plt.ylabel("Cummulative Sum Returns (%)")
plt.xlabel(None)
plt.tight_layout()
plt.savefig(results_folder / "monthly_momentum_sum_bt.png")
plt.clf()

# Cummulative Product Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=pnl, x="date", y="cumprod")
plt.ylabel("Cummulative Product Returns (%)")
plt.xlabel(None)
plt.tight_layout()
plt.savefig(results_folder / "monthly_momentum_prod_bt.png")
plt.clf()
