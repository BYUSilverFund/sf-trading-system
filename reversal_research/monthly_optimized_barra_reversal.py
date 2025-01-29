from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.reversal_z_strategy import ReversalZStrategy
from silverfund.datasets import MasterMonthly

# Monthly backtest
start_date = date(1995, 7, 31)  # Russell is missing 2012-07, and a day early 2017-11-29
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = (
    MasterMonthly(
        start_date=start_date,
        end_date=end_date,
    )
    .load_all()
    .select(["date", "barrid", "mktcap", "div_yield", "price", "ret", "total_risk", "spec_risk", "log_spec_ret"])
)

# Create backtest instance
bt = Backtester(start_date=start_date, end_date=end_date, interval=Interval.MONTHLY, historical_data=historical_data, strategy=MomentumZStrategy, security_identifier="barrid")

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)

min_date = pnl["date"].min().strftime("%Y-%m-%d")
max_date = pnl["date"].max().strftime("%Y-%m-%d")

print(f"From {min_date} to {max_date}")
print(pnl)

results_folder = "/Users/bwaits/Research/sf-trading-system/reversal_research"

pnl.write_parquet(f"{results_folder}/monthly_optimized_reversal_bt.parquet")
