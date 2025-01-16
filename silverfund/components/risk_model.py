#   Module: risk_model.py
#   Author: Brandon Bates and Seth Peterson
#   Date: May 3 2023
#   Purpose: Compute the Asset Covariance Matrix using factor exposures.
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - RiskModel Abstract Base Class
#   - FactorRiskModel Class
#
# ------------------------------------------------------------------------------------------------ #

# --- Import Modules ---
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
import tqdm

from silverfund.datasets.dataio import (load_default_monthly_universe,
                                        load_exposure_matrix,
                                        load_factor_covariance_matrix,
                                        load_idio_risk_vector)

# ----------------------------------------------------------------------------------------------
#                                    Risk Model Classes
# ----------------------------------------------------------------------------------------------

# -----------------------------------------
#          Base RiskModel Class
# -----------------------------------------


class RiskModel(ABC):

    @abstractmethod
    def get_valid_ids(self) -> list[str]:
        pass

    @abstractmethod
    def to_df(self, ids: list[str] | pd.Index = None) -> pd.DataFrame:
        pass


# -----------------------------------------
#           FactorRiskModel Class
# -----------------------------------------
class FactorRiskModel(RiskModel):

    def __init__(
        self,
        factor_exposures: pd.DataFrame,
        factor_cov: pd.DataFrame,
        idio_risk: pd.DataFrame,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        factor_exposures : pd.DataFrame
            exposures to the Barra factors, dim (N stocks by k factors)
        factor_cov : pd.DataFrame
            covariance matrix of factor returns, dim (k factors x k factors)
        idio_risk : pd.DataFrame
            vector of idiosyncratic volatilities, dim (N stocks x 1)
        verbose : bool, optional
            whether to print intermediate results, by default False
        """
        # TODO: Determine whether the date Barra uses corresponds to the date of the data used for the calculations
        #  or the data you could have reasonably traded on it.  (Do we need to lag the risk model a couple days?)
        self.factor_exposures = factor_exposures
        self.factor_cov = factor_cov
        self.idio_risk = idio_risk
        self.risk_model = None

        # assemble components for the full risk model
        self._calc_risk_model()  # calcs the self.risk_model field

        # metadata attributes
        self.asset_ids = self.factor_exposures.index
        self.factor_ids = self.factor_exposures.columns

    # --- Public Methods ---
    def get_valid_ids(self) -> list[str]:
        row_entries_that_are_all_nans = self.risk_model.index[self.risk_model.isna().all(axis=1)]
        assets_without_all_nans = self.risk_model.index.difference(row_entries_that_are_all_nans)
        valid_asset_risk_model = self.risk_model.loc[
            assets_without_all_nans, assets_without_all_nans
        ]
        # check for assets with only some missing values
        valid_assets_with_missing_values = valid_asset_risk_model.notna().all(
            axis=1
        ) | valid_asset_risk_model.notna().all(axis=0)
        return valid_asset_risk_model.index[valid_assets_with_missing_values].to_list()

    def to_df(self, ids: list[str] | pd.Index = None) -> pd.DataFrame:
        if ids is not None:
            return self.risk_model.loc[ids, ids]
        else:
            return self.risk_model

    def get_variance_vector(self) -> pd.Series:
        return pd.Series(np.diag(self.risk_model), index=self.risk_model.index)

    # --- Private Methods ---
    def _calc_risk_model(self) -> None:
        """
        Calculate the full risk model, which is the sum of the common covariance and idiosyncratic variance
        """
        idio_risk = np.power(self.idio_risk, 2)
        idio_risk = pd.DataFrame(
            index=idio_risk.index,
            columns=idio_risk.index,
            data=np.diag(idio_risk.to_numpy().flatten()),
        )
        self.risk_model = (
            self.factor_exposures @ self.factor_cov @ self.factor_exposures.T + idio_risk
        )

    # --- Magic Methods ---
    def __repr__(self):
        return self.risk_model.__repr__()

    # --- Class Methods ---
    @classmethod
    def load(
        cls,
        date: str | pd.Timestamp,
        barrids: list[str] | pd.Index = None,
        factorids: list[str] | pd.Index = None,
        omitted_factors: str | list[str] | pd.Index = None,
    ) -> Self:
        if type(omitted_factors) == str:
            omitted_factors = [omitted_factors]
        factor_exposures = load_exposure_matrix(
            date=date, barrids=barrids, factorids=factorids, omitted_factors=omitted_factors
        )
        factor_cov_matrix = load_factor_covariance_matrix(
            date=date, factorids=factorids, omitted_factors=omitted_factors
        )
        idio_risk = load_idio_risk_vector(date=date, barrids=barrids)
        return cls(
            factor_exposures=factor_exposures, factor_cov=factor_cov_matrix, idio_risk=idio_risk
        )


class RiskModelGenerator:

    @classmethod
    def get_vols_over_range(cls, start_date: pd.Timestamp, end_date: pd.Timestamp):
        data = {}
        for date in tqdm.tqdm(pd.date_range(start_date, end_date)):
            try:
                rm = FactorRiskModel.load(date)
                data[date] = rm.get_variance_vector()
            except:
                continue
        return pd.DataFrame(data)
