from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.datasets.crsp_monthly import CRSPMonthly
from silverfund.enums import Interval
from silverfund.strategies.momentum_strategy import MomentumStrategy

# Monthly backtest
start_date = date(1995, 7, 31)
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = (
    CRSPMonthly(
        start_date=start_date,
        end_date=end_date,
    )
    .load_all()
    .select(["date", "permno", "ticker", "ret", "prc"])
)

# Create backtest instance
bt = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=MomentumStrategy,
    security_identifier="permno",
)

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)

min_date = pnl["date"].min().strftime("%Y-%m-%d")
max_date = pnl["date"].max().strftime("%Y-%m-%d")

print(f"From {min_date} to {max_date}")

print(pnl)

# Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=pnl, x="date", y="cumsum")
plt.ylabel("Cummulative Sum Returns (%)")
plt.xlabel(None)
plt.tight_layout()

results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/classic_momentum/results"

pnl.write_parquet(f"{results_folder}/monthly_momentum_bt.parquet")
plt.savefig(f"{results_folder}/monthly_momentum_bt.png")
