from datetime import date
from functools import partial

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import full_investment
from silverfund.optimizers import quadratic_program
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

start_date = date(2020, 1, 1)
end_date = date(2024, 12, 31)

strat = Strategy(
    signal_fn=momentum,
    score_fn=partial(z_score, col="mom", over="barrid"),
    alpha_fn=grindold_kahn,
    optimizer=quadratic_program,
    constraints=[full_investment],
)

bt = Backtester(start_date=start_date, end_date=end_date, strategy=strat)

pnl = bt.run()

print(pnl)
