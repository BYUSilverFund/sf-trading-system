#   Module: alpha.py
#   Author: Brandon Bates and Seth Peterson
#   Date: Apr 2023
#   Purpose: Representation and computation of alphas.
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - Alpha Abstract Base Class
#   - PreComputedAlpha Class
#   - AlphaVsBenchmark Class
#
# ------------------------------------------------------------------------------------------------ #

# --- Import Modules ---
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from silverfund.components.risk_model import RiskModel

# ----------------------------------------------------------------------------------------------
#                                     Alpha Classes
# ----------------------------------------------------------------------------------------------


# -----------------------------------------
#          Base Alpha Class
# -----------------------------------------
class Alpha(ABC):

    _ids = None

    @abstractmethod
    def get_valid_ids(self) -> list[str]:
        pass

    @abstractmethod
    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        pass

    # --- Magic Methods ---
    def __repr__(self):
        return self.alpha.__repr__()

    # --- Properties ---
    @property
    def ids(self):
        return self._ids


# -----------------------------------------
#         Pre-Computed Alpha Class
# -----------------------------------------
class PreComputedAlpha(Alpha):
    """Takes a panel of alphas the researcher has computed elsewhere."""

    def __init__(self, precomputed_alpha: pd.DataFrame, *args, **kwargs) -> None:
        self._alpha = None
        self.alpha = precomputed_alpha

    def get_valid_ids(self) -> list[str]:
        return self.alpha.columns[self.alpha.notna().all()].to_list()

    def to_df(self, ids: list[str] | pd.Index = None) -> pd.DataFrame:
        if ids is not None:
            return self.alpha.loc[:, ids]
        else:
            return self.alpha

    # --- Properties ---
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha: pd.DataFrame):
        if not isinstance(new_alpha, pd.DataFrame):
            raise TypeError("Input alpha must be of type pd.DataFrame.")
        if new_alpha.shape[0] > new_alpha.shape[1]:
            print(new_alpha.shape)
            raise ValueError("Input alpha must be a row vector (1 x N), not a column vector.")
        self._alpha = new_alpha
        self._ids = new_alpha.columns.to_list()

    @property
    def ids(self):
        return self._ids


# -----------------------------------------
#       Grinold and Kahn Alpha Class
# -----------------------------------------
class GrinoldKahnAlpha(Alpha):
    """Use the G&K formula to convert scores, vols, and ICs into alphas."""

    def get_valid_ids(self) -> list[str]:
        pass

    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        pass
