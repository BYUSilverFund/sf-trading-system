from datetime import date

from silverfund.backtester import Backtester
from silverfund.components.enums import Interval
from silverfund.components.strategies.reversal_strategy import ReversalStrategy

bt = Backtester(
    start_date=date(2020, 9, 1),
    end_date=date.today(),
    interval=Interval.DAILY,
    strategy=ReversalStrategy,
)
bt.run()
