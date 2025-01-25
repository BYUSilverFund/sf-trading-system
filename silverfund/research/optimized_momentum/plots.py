import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

pnl = pl.read_parquet("silverfund/research/optimized_momentum/results/monthly_optimized_momentum_bt.parquet")

# Calculate sharpe ratio
portfolio_er = pnl.select("portfolio_ret").mean()
portfolio_vol = pnl.select("portfolio_ret").std()

annual_factor = 12 / np.sqrt(12)

sharpe = (portfolio_er / portfolio_vol) * annual_factor

print(f"Annualized Sharpe:", round(sharpe.item(), 4))

# Cummulative Product Chart
plt.figure(figsize=(10, 6))
sns.lineplot(pnl, x="date", y="cumprod")
plt.xlabel(None)
plt.ylabel("Cummulative Product Returns (%)")

# Save and clear
results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/optimized_momentum/results"
plt.savefig(f"{results_folder}/monthly_optimized_momentum_product_bt.png")
plt.clf()

# Cummulative Sum Chart
plt.figure(figsize=(10, 6))
sns.lineplot(pnl, x="date", y="cumsum")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Returns (%)")

# Save and clear
results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/optimized_momentum/results"
plt.savefig(f"{results_folder}/monthly_optimized_momentum_sum_bt.png")
plt.clf()
