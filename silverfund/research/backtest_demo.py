#   Module: backtest_demo.py
#   Author: Brandon Bates and Seth Peterson
#   Date: October 30, 2023
#   Purpose: Demonstrate how to use the backtester.
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - Demo of backtester
# ------------------------------------------------------------------------------------------------ #

# --- Imports ---
import numpy as np
import pandas as pd
import polars as pl

from silverfund.components.alpha import PreComputedAlpha
from silverfund.components.optimizers import MVPortfolioConstructor
from silverfund.components.optimizers.constraints import *
from silverfund.components.risk_model import FactorRiskModel
from silverfund.datasets.dataio import *

# ------------------------------------------------------------------------------------------------ #
# Backtest Dates and Universe
n_assets = 2000
date = "1995-07-10"
dt = pd.to_datetime(date)
spec_risk = load_idio_risk_vector(date=date)
univ = load_default_monthly_universe()
univ_dt = univ.loc[(univ.index.year == dt.year) & (univ.index.month == dt.month)]
univ_dt = univ_dt.columns[univ_dt.all(axis=0)]
valid_barra_ids = spec_risk.loc[spec_risk.notna().values].index
valid_barra_ids = valid_barra_ids.intersection(univ_dt)[-n_assets:]

# ----------------------------------------------------------------------------------------------------------------
# --- Single Period Optimization with Constraints ---
frm = FactorRiskModel.load(date=date, barrids=valid_barra_ids)

alpha = PreComputedAlpha(
    precomputed_alpha=pd.DataFrame(
        index=[dt], columns=valid_barra_ids, data=np.ones(shape=(1, n_assets))
    )
)
cons = [ZeroHoldingConstraint(asset_ids_to_zero=valid_barra_ids[2:4]), FullInvestmentConstraint()]

mvpc = MVPortfolioConstructor(date=dt, alpha=alpha, risk_model=frm, constraints=cons)
pp = mvpc.get_optimal_portfolio()

print("Single Period Portfolio", pl.from_pandas(pp))

# ----------------------------------------------------------------------------------------------------------------
# --- History of Optimal Portfolios ---
from silverfund.backtester import MVBacktester

start_dt_ix = 1
end_dt_ix = 20

valid_barra_dates = pd.to_datetime(load_list_of_valid_barra_dates())

cons = [FullInvestmentConstraint()]

alpha = pd.DataFrame(
    index=valid_barra_dates[start_dt_ix:end_dt_ix], columns=valid_barra_ids, data=1
)
bt = MVBacktester(alpha=alpha, constraints=cons)
result = bt.get_optimal_portfolio_history(n_cpus=4)

print("Backtest History", pl.from_pandas(result))
