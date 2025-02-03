#   Module: mv_optimizer.py
#   Author: Brandon Bates
#   Date: November 2023
#   Purpose: Solve for MVE portfolio weights
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - MVPortfolioConstructor Class
# ------------------------------------------------------------------------------------------------ #

# --- Import Modules ---
from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pandas as pd

from silverfund.components.alpha import Alpha
from silverfund.components.risk_model import RiskModel
from silverfund.optimizers.constraints import Constraint

# ------------------------------------------------------------------------------------------------ #
#                                 MVPortfolioConstructor Class
# ------------------------------------------------------------------------------------------------ #


class MVPortfolioConstructor:

    def __init__(
        self,
        date: pd.Timestamp,
        alpha: Alpha,
        risk_model: RiskModel,
        constraints: list[Constraint] = None,
        risk_aversion_coefficient: float = 2,
    ):
        self.date = date
        self.alpha = alpha
        self.risk_model = risk_model
        self.constraints = constraints if constraints is not None else []
        self.risk_aversion = risk_aversion_coefficient

        self._lagrange_multipliers = None
        self._optimal_portfolio = None
        self._problem = None
        self._solved = False
        self._valid_ids = self._find_valid_assets()

    # --- Public Methods ---
    def get_lagrange_multipliers(self) -> dict:
        if not self._solved:
            self._solve()
        return self._lagrange_multipliers

    def get_optimal_portfolio(self) -> pd.DataFrame:
        if not self._solved:
            self._solve()
        return self._optimal_portfolio

    # --- Private Methods ---
    def _find_valid_assets(self) -> list[str]:
        valid_alpha = set(self.alpha.get_valid_ids())
        valid_risk = set(self.risk_model.get_valid_ids())
        valid_ids = valid_alpha.intersection(valid_risk)

        # find and intersect with valid ids from constraints
        if len(self.constraints) > 0:
            for con in self.constraints:
                v_ids = con.get_valid_ids(dt=self.date)
                if v_ids is not None:
                    valid_ids.intersection(v_ids)

        return list(valid_ids)

    def _solve(self) -> None:
        # set up the optimization problem
        w = cvx.Variable(len(self._valid_ids))
        mv_utility_fn = w @ self.alpha.to_df(
            self._valid_ids
        ).T - 0.5 * self.risk_aversion * cvx.quad_form(
            x=w, P=self.risk_model.to_df(self._valid_ids)
        )
        obj_fn = cvx.Maximize(mv_utility_fn)
        cons_set = [
            cons.construct(wt=w, valid_ids=self._valid_ids, dt=self.date)
            for cons in self.constraints
        ]
        self._problem = cvx.Problem(objective=obj_fn, constraints=cons_set)

        # find the solution
        self._problem.solve()

        # unpack the results
        self._optimal_portfolio = self._package_optimal_portfolio(w)
        self._lagrange_multipliers = self._package_lagrange_multipliers()

        # change solution state
        self._solved = True

    def _package_optimal_portfolio(self, wt: cvx.Variable) -> pd.DataFrame:
        # first create portfolio with just the assets that had valid data for all components
        port = pd.DataFrame(index=self._valid_ids, columns=[self.date], data=wt.value).T

        # augment the data frame with additional ids for which there were missing values
        additional_ids = list(set(self.alpha.ids).difference(port.columns))
        if len(additional_ids) > 0:
            # Make a dataframe with the additional ids and fill with NaNs
            additional_port = pd.DataFrame(index=[self.date], columns=additional_ids, data=np.nan)
            # Concatenate the two dataframes
            port = pd.concat([port, additional_port], axis=1)

        return port

    def _package_lagrange_multipliers(self) -> dict:
        cons_names = [cons.name for cons in self.constraints]
        lm_vals = list(self._problem.solution.dual_vars.values())
        return dict(zip(cons_names, lm_vals))
