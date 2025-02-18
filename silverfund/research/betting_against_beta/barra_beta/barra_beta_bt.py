from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from silverfund.backtester import Backtester
from silverfund.datasets.master_monthly import MasterMonthly
from silverfund.enums import Interval
from silverfund.optimizers.new_constraints import *
from silverfund.research.betting_against_beta.barra_beta.barra_beta_strategy import BarraBetaStrategy

# Daily backtest
start_date = date(2022, 1, 1)
end_date = date(2024, 12, 31)

# Load historical dataset
historical_data = (
    MasterMonthly(
        start_date=start_date,
        end_date=end_date,
    )
    .load_all()
)

# Create backtest instance
bt = Backtester(
    start_date=start_date,
    end_date=end_date,
    interval=Interval.MONTHLY,
    historical_data=historical_data,
    strategy=BarraBetaStrategy,
    constraints=[FullInvestment, NoBuyingOnMargin, ShortingLimit, UnitBeta],
    security_identifier="barrid"
)

# Run backtest
pnl = bt.run()

# Table
print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)
print(pnl)

# Save
results_folder = "/Users/jaic007/src/sf-trading-system/silverfund/research/betting_against_beta/barra_beta/results"
pnl.write_parquet(f"{results_folder}/daily_momentum_bt.parquet")
