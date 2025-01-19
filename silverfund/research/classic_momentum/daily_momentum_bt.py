from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.momentum_strategy import MomentumStrategy
from silverfund.datasets import CRSPDaily

# Daily backtest
start_date = date(2022, 1, 1)
end_date = date(2024, 8, 31)

# Load historical dataset
historical_data = (
    CRSPDaily(
        start_date=start_date,
        end_date=end_date,
    )
    .load_all()
    .select(["date", "permno", "ret"])
)

# Create backtest instance
bt = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.DAILY,
    historical_data=historical_data,
    strategy=MomentumStrategy,
)

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Chart
sns.lineplot(data=pnl, x="date", y="cumsum")
plt.ylabel("Cummulative returns (sum)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
