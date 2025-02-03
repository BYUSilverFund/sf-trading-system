from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.datasets.crsp_daily import CRSPDaily
from silverfund.enums import Interval
from silverfund.strategies.momentum_strategy import MomentumStrategy

# Daily backtest
start_date = date(2022, 1, 1)
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = (
    CRSPDaily(
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
    interval=Interval.DAILY,
    historical_data=historical_data,
    strategy=MomentumStrategy,
    security_identifier="permno",
)

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/classic_momentum/results"
pnl.write_parquet(f"{results_folder}/daily_momentum_bt.parquet")
