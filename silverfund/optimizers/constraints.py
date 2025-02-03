#   Module: constraints.py
#   Author: Brandon Bates
#   Date: October 2023
#   Purpose: Representation and computation of constraints.
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - Constraint Class
#   - ZeroInvestmentConstraint Class
#   - FullInvestmentConstraint Class
#   - BetaToBenchmarkConstraint Class
#
# ------------------------------------------------------------------------------------------------ #

# --- Import Modules ---
from __future__ import annotations

from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
#                                     Constraint Classes
# ----------------------------------------------------------------------------------------------


# -----------------------------------------
#       Base Constraint Class
# -----------------------------------------
class Constraint(ABC):

    name = "unnamed"

    @abstractmethod
    def get_valid_ids(self, dt: str | pd.Timestamp = None) -> list[str] | None:
        pass

    @abstractmethod
    def construct(
        self, wt: cvx.Variable, dt: str | pd.Timestamp = None, **kwargs
    ) -> cvx.Constraint:
        pass


# -----------------------------------------
#       ZeroInvestmentConstraint Class
# -----------------------------------------
class ZeroInvestmentConstraint(Constraint):

    name = "zero_investment"

    def get_valid_ids(self, dt: str | pd.Timestamp = None) -> list[str] | None:
        return None

    def construct(
        self, wt: cvx.Variable, dt: str | pd.Timestamp = None, **kwargs
    ) -> cvx.Constraint:
        return cvx.sum(wt) == 0


# -----------------------------------------
#       FullInvestmentConstraint Class
# -----------------------------------------
class FullInvestmentConstraint(Constraint):

    name = "full_investment"

    def get_valid_ids(self, dt: str | pd.Timestamp = None) -> list[str] | None:
        return None

    def construct(
        self, wt: cvx.Variable, dt: str | pd.Timestamp = None, **kwargs
    ) -> cvx.Constraint:
        return cvx.sum(wt) == 1


# -----------------------------------------
#        ZeroHoldingConstraint Class
# -----------------------------------------
class ZeroHoldingConstraint(Constraint):

    name = "zero_holding"

    def __init__(self, asset_ids_to_zero: list[str]) -> None:
        self._asset_ids_to_zero = asset_ids_to_zero

    def get_valid_ids(self, dt: str | pd.Timestamp = None) -> list[str] | None:
        return None

    def construct(
        self, wt: cvx.Variable, valid_ids: list[str], dt: str | pd.Timestamp = None, **kwargs
    ) -> cvx.Constraint:

        assets_in_univ_to_zero = [aid for aid in self._asset_ids_to_zero if aid in valid_ids]
        ix_of_asset = [valid_ids.index(aid) for aid in assets_in_univ_to_zero]

        cons_mat = np.zeros(shape=(len(assets_in_univ_to_zero), len(valid_ids)))
        for row, col in enumerate(ix_of_asset):
            cons_mat[row, col] = 1

        return cons_mat @ wt == 0


# -----------------------------------------------------
#        PortfolioFactorExposureConstraint Class
# -----------------------------------------------------
class PortfolioFactorExposureConstraint(Constraint):

    name = "portfolio_factor_exposure"

    def __init__(self, factor_exposures: pd.DataFrame, exposure_target: float = 0) -> None:
        self._factor_exposures = factor_exposures
        self._exposure_target = exposure_target

    def get_valid_ids(self, dt: str | pd.Timestamp = None) -> list[str]:
        # Get the valid asset ids for the given date
        df_dt = self._factor_exposures.loc[dt, :]
        mask = df_dt.isna()
        # Return the asset ids that are not missing
        return list(df_dt.index[~mask])

    def construct(
        self, wt: cvx.Variable, valid_ids: list[str], dt: str | pd.Timestamp = None, **kwargs
    ) -> cvx.Constraint:

        fact_exp_dt = self._factor_exposures.loc[dt, valid_ids].to_frame().T
        if fact_exp_dt.shape[1] != wt.shape[0]:
            raise ValueError(
                f"The factor exposure vector must be the same size as the portfolio vector.\
                             Shapes: {fact_exp_dt.shape, wt.shape}"
            )
        return wt.T @ np.squeeze(fact_exp_dt) == self._exposure_target


# -----------------------------------------------------
#        ShortSellingConstraint Class
# -----------------------------------------------------
class ShortSellingConstraintClass(Constraint):

    name = "short_selling"

    def get_valid_ids(self, dt: str | pd.Timestamp = None) -> list[str] | None:
        return None

    def construct(
        self, wt: cvx.Variable, dt: str | pd.Timestamp = None, **kwargs
    ) -> cvx.Constraint:
        return wt >= 0
