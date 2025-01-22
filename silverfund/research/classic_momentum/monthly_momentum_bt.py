from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.momentum_strategy import MomentumStrategy
from silverfund.datasets import CRSPMonthly

# Monthly backtest
start_date = date(2006, 1, 1)
end_date = date(2024, 8, 31)

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
plt.show()
