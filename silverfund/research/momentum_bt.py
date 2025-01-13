from datetime import date

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.momentum_strategy import MomentumStrategy

print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)


bt = Backtester(
    start_date=date(2020, 9, 1),
    end_date=date.today(),
    interval=Interval.DAILY,
    strategy=MomentumStrategy,
)
bt.run()
