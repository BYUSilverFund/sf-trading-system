#   Module: backtester.py
#   Author: Brandon Bates
#   Date: October 2023
#   Purpose: Class to perform historical optimal portfolio simulations.
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - Backtester Class
#
# ------------------------------------------------------------------------------------------------ #

# --- Import Modules ---
import pandas as pd
import ray
from ray.experimental import tqdm_ray
from tqdm import tqdm

import silverfund.components.optimizers.constraints as cons
from silverfund.components.alpha import PreComputedAlpha
from silverfund.components.optimizers import MVPortfolioConstructor
from silverfund.components.risk_model import FactorRiskModel
from silverfund.datasets.dataio import *

# ----------------------------------------------------------------------------------------------
#                                       MVBacktester Class
# ----------------------------------------------------------------------------------------------


class MVBacktester:
    def __init__(
        self,
        alpha: pd.DataFrame,
        univ: pd.DataFrame = None,
        constraints: list[cons.Constraint] = None,
        risk_aversion_coefficient: float = 2,
        omitted_factors: str | list[str] = None,
    ):
        self._alpha = None
        self._univ = None
        self._risk_aversion = risk_aversion_coefficient
        self._optimal_portfolio_history = None
        self._optimized = False
        self._n_cpus = 1
        self.univ = univ
        self.omitted_factors = omitted_factors

        # attributes set by properties
        self.alpha = alpha
        self.constraints = constraints

        # derived attributes
        self._valid_dates: pd.DatetimeIndex = self._find_valid_dates()

    # --- Public Methods ---
    def get_optimal_portfolio_history(self, n_cpus: int = None) -> pd.DataFrame:
        if n_cpus is not None:
            self._n_cpus = n_cpus
        if not self._optimized:
            if self._n_cpus == 1:
                self._calc_opt_port_hist_single_thread()
            else:
                self._calc_opt_port_hist_parallel()
            self._optimized = True
        return self._optimal_portfolio_history

    # --- Private Methods ---
    def _calc_opt_port_hist_single_thread(self) -> None:
        opt_port_list = []
        for dt in tqdm(self._valid_dates):
            mvpc = self._create_portfolio_constructor(dt)
            try:
                opt_port_dt = mvpc.get_optimal_portfolio()
                opt_port_list.append(opt_port_dt)
            except Exception as e:
                print("Error:", e)
        self._optimal_portfolio_history = (
            pd.concat(opt_port_list, axis=0).sort_index(axis=0).sort_index(axis=1)
        )

    def _calc_opt_port_hist_parallel(self) -> None:
        n_cpus = min(self._n_cpus, len(self._valid_dates))
        ray.init(ignore_reinit_error=True, num_cpus=n_cpus)
        oph_ray = ray.put(self)

        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        progress_bar = remote_tqdm.remote(total=len(self._valid_dates))

        # build computation strategy
        ray_actors = []
        for dt_ in self._valid_dates:
            ray_actors.append(
                self._opt_port_worker.remote(obj=oph_ray, dt=dt_, progress_bar=progress_bar)
            )

        # execute optimization
        opt_port_collection = ray.get(ray_actors)

        progress_bar.close.remote()
        ray.shutdown()

        # write output
        self._optimal_portfolio_history = (
            pd.concat(opt_port_collection, axis=0).sort_index(axis=0).sort_index(axis=1)
        )

    @staticmethod
    @ray.remote
    def _opt_port_worker(obj, dt: pd.Timestamp, progress_bar: tqdm_ray.tqdm):

        # build portfolio constructor
        univ_dt = obj._get_universe(dt)
        valid_univ_dt = univ_dt.intersection(obj.alpha.columns)
        try:
            risk_model = FactorRiskModel.load(
                date=dt, barrids=valid_univ_dt, omitted_factors=obj.omitted_factors
            )
        except KeyError as e:
            key = str(e).split("'")[1]
            print(f"Error: {key} not found in risk model data for date {dt}.")
            return None
        alpha_dt = PreComputedAlpha(precomputed_alpha=obj.alpha.loc[dt, valid_univ_dt].to_frame().T)

        mvpc = MVPortfolioConstructor(
            date=dt,
            alpha=alpha_dt,
            risk_model=risk_model,
            constraints=obj.constraints,
            risk_aversion_coefficient=obj._risk_aversion,
        )

        # run the optimization
        progress_bar.update.remote(1)
        # Skip errors
        try:
            port = mvpc.get_optimal_portfolio()
        except Exception as e:
            # Error
            print("Error:", e)
            return None
        return port

    def _create_portfolio_constructor(self, dt: pd.Timestamp) -> MVPortfolioConstructor:
        valid_univ_dt = self._get_valid_universe(dt)
        alpha_dt = PreComputedAlpha(
            precomputed_alpha=self.alpha.loc[dt, valid_univ_dt].to_frame().T
        )
        risk_model = FactorRiskModel.load(
            date=dt, barrids=valid_univ_dt, omitted_factors=self.omitted_factors
        )

        return MVPortfolioConstructor(
            date=dt,
            alpha=alpha_dt,
            risk_model=risk_model,
            constraints=self.constraints,
            risk_aversion_coefficient=self._risk_aversion,
        )

    def _find_valid_dates(self) -> pd.DatetimeIndex:
        valid_barra_dates = pd.to_datetime(load_list_of_valid_barra_dates())
        alpha_dates = self.alpha.index
        return valid_barra_dates.intersection(alpha_dates)

    def _get_universe(self, dt: pd.Timestamp) -> pd.Index:
        """Extracts the universe for the given date."""
        # TODO: Figure out if the universe df is monthly or daily, and extract current universe accordingly.
        # is_monthly_univ = True
        # if is_monthly_univ:
        mth_index = (self.univ.index.year == dt.year) & (self.univ.index.month == dt.month)
        univ_dt = self.univ.loc[mth_index]
        return univ_dt.columns[univ_dt.all(axis=0)]

    def _get_valid_universe(self, dt: pd.Timestamp):
        """Gets the non-NaN alphas at the given date."""
        dt_univ = self._get_universe(dt)
        dt_alpha = self.alpha.loc[dt, :]
        dt_alpha = dt_alpha.loc[dt_alpha.notna()]
        return dt_univ.intersection(dt_alpha.index)

    # --- Properties ---
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha: pd.DataFrame):
        # verify index is a datetime index
        if not isinstance(new_alpha.index, pd.DatetimeIndex):
            raise TypeError(
                "The 'alpha' argument must be a pd.DataFrame with a pd.DatetimeIndex for its index."
            )
        self._alpha = new_alpha

    @property
    def univ(self):
        return self._univ

    @univ.setter
    def univ(self, new_univ: pd.DataFrame):
        if new_univ is not None:
            # verify index is a datetime index
            if not isinstance(new_univ.index, pd.DatetimeIndex):
                raise TypeError(
                    "The 'univ' argument must be a pd.DataFrame with a pd.DatetimeIndex for its index."
                )
        else:  # load default universe from infrastructure
            new_univ = load_default_monthly_universe()
        self._univ = new_univ
