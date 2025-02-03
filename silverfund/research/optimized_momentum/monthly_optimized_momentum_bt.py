from datetime import date

from silverfund.backtester import Backtester
from silverfund.datasets.master_monthly import MasterMonthly
from silverfund.enums import Interval
from silverfund.optimizers.new_constraints import *
from silverfund.strategies.momentum_z_strategy import MomentumZStrategy

# Monthly backtest
start_date = date(2010, 1, 1)  # date(1998, 10, 31)
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = MasterMonthly(
    start_date=start_date,
    end_date=end_date,
).load_all()

# Create backtest instance
bt = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=MomentumZStrategy,
    constraints=[FullInvestment, NoLeverage, LongOnly, UnitBeta],
    security_identifier="barrid",
)

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
results_folder = "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/optimized_momentum/results"
pnl.write_parquet(f"{results_folder}/monthly_optimized_momentum_bt.parquet")
