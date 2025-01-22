from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.momentum_z_strategy import MomentumZStrategy
from silverfund.datasets import MasterMonthly

# Daily backtest
start_date = date(2011, 6, 1)  # date(2007, 6, 1)
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = (
    MasterMonthly(
        start_date=start_date,
        end_date=end_date,
    )
    .load_all()
    .select(["date", "barrid", "mktcap", "price", "ret", "total_risk", "spec_risk"])
)

# Create backtest instance
bt = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=MomentumZStrategy,
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
sns.lineplot(data=pnl, x="date", y="cumsum")
plt.ylabel("Cummulative Returns Sum (%)")
plt.xlabel(None)
plt.tight_layout()

results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/optimized_momentum/results"

pnl.write_parquet(f"{results_folder}//monthly_optimized_momentum_bt.parquet")
plt.savefig(f"{results_folder}/monthly_optimized_momentum_bt.png")
