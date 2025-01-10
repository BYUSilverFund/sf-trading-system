from datetime import date
from functools import partial

from backtester import Backtester
from strategies import momentum_strategy

print("\n" + "-" * 50 + " Last Period Portfolio " + "-" * 50)

portfolios = momentum_strategy(interval="monthly")
print(portfolios[-1])

print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)

bt = Backtester(
    start_date=date(2020, 1, 1),
    end_date=date(2024, 12, 31),
    interval="monthly",
    strategy=partial(momentum_strategy, interval="monthly"),
)
bt.run()
