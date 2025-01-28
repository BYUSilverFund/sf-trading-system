from datetime import date

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.momentum_z_strategy import MomentumZStrategy
from silverfund.datasets.master_monthly import MasterMonthly

# Monthly backtest
start_date = date(1995, 7, 31)
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
    security_identifier="barrid",
)

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/optimized_momentum/results"
pnl.write_parquet(f"{results_folder}//monthly_optimized_momentum_bt.parquet")
