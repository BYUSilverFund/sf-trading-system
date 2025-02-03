import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

results_folder = (
    "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/constraints/results"
)


def calc_sharpe(returns: pl.Series):
    port_ret = returns.mean()
    port_vol = returns.std()
    annual_factor = 12 / np.sqrt(12)
    return (port_ret / port_vol) * annual_factor


constraints = ["base", "long_only", "unit_beta"]
pnls = []
sharpes = {}
for file in constraints:
    pnl = pl.read_parquet(results_folder + f"/{file}_bt.parquet")
    pnl = pnl.with_columns(pl.lit(file).alias("constraint"))
    pnls.append(pnl)

    sharpe = calc_sharpe(pnl["portfolio_ret"])
    sharpes[file] = sharpe

pnl: pl.DataFrame = pl.concat(pnls)

pnl = pnl.with_columns(
    pl.col("constraint").map_elements(lambda x: f'{x.replace("_", " ").title()}: {sharpes[x]:.2f}')
)

# Cummulative Sum Chart
plt.figure(figsize=(10, 6))
sns.lineplot(pnl, x="date", y="cumsum", hue="constraint")
plt.title("Optimized Monthly Momentum")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Returns (%)")
plt.legend(title="Constraint: Sharpe")

# Save and clear
plt.savefig(f"{results_folder}/constraints_demo.png", dpi=300)
plt.clf()
