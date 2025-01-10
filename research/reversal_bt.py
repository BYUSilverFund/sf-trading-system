from datetime import date
from functools import partial

from backtester import Backtester
from strategies import reversal_strategy

print("\n" + "-" * 50 + " Last Period Portfolio " + "-" * 50)

portfolios = reversal_strategy(interval="daily")
print(portfolios[-1])


print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)

bt = Backtester(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    interval="daily",
    strategy=partial(reversal_strategy, interval="daily"),
)
bt.run()
