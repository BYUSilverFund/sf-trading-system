from datetime import date

from silverfund.backtester import Backtester
from silverfund.datasets.master_monthly import MasterMonthly
from silverfund.enums import Interval
from silverfund.optimizers.new_constraints import *
from silverfund.strategies.momentum_z_strategy import MomentumZStrategy

results_folder = (
    "/Users/andrew/Projects/SilverFund/sf-trading-system/silverfund/research/constraints/results"
)

# Monthly backtest
start_date = date(2020, 1, 1)
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = MasterMonthly(
    start_date=start_date,
    end_date=end_date,
).load_all()

# Weights between -1 and 1 that sum to 1
base_bt = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=MomentumZStrategy,
    constraints=[FullInvestment, NoBuyingOnMargin, ShortingLimit],
    security_identifier="barrid",
)

# Run backtest
pnl = base_bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
pnl.write_parquet(f"{results_folder}/base_bt.parquet")

# Weights between 0 and 1 that sum to 1
long_only = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=MomentumZStrategy,
    constraints=[FullInvestment, NoBuyingOnMargin, LongOnly],
    security_identifier="barrid",
)

# Run backtest
pnl = long_only.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
pnl.write_parquet(f"{results_folder}/long_only_bt.parquet")

# Weights between 0 and 1 that sum to 1 and have a benchmark beta of 1
unit_beta = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=MomentumZStrategy,
    constraints=[FullInvestment, NoBuyingOnMargin, LongOnly, UnitBeta],
    security_identifier="barrid",
)

# Run backtest
pnl = unit_beta.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
pnl.write_parquet(f"{results_folder}/unit_beta_bt.parquet")
