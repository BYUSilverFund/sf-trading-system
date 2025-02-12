from datetime import date

import polars as pl

from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import full_investment
from silverfund.optimizers import quadratic_program
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

start_date = date(2020, 1, 1)
end_date = date(2024, 12, 31)


class Vols(pl.DataFrame):
    pass


strat = Strategy(
    signal_fn=momentum,
    score_fn=z_score,
    alpha_fn=grindold_kahn,
    optimizer=quadratic_program,
    constraints=[full_investment],
)

bt = Backtester(start_date=start_date, end_date=end_date, strategy=strat)

pnl = bt.run()

print(pnl)
