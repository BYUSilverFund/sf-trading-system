from datetime import date

import numpy as np
import polars as pl

from silverfund.datasets import BarraFactorCovariances, BarraFactorExposures, BarraSpecificRiskForecast, Master


class NewRiskModel:

    def __init__(self, date_: date) -> None:
        self._date = date_
        self._get_unique_universe()

    def _get_unique_universe(self) -> pl.DataFrame:
        # Load all necessary datasets
        master = Master(self._date, self._date).load_all()
        bfe = BarraFactorExposures().load(self._date.year, self._date)
        bsrf = BarraSpecificRiskForecast().load(self._date.year, self._date)

        # Get each set of unique barrids
        master_ids = master.select("barrid").unique()
        bfe_ids = bfe.select("barrid").unique()
        bsrf_ids = bsrf.select("barrid").unique()

        # Find the ineer join of all id lists
        universe = master_ids.join(bfe_ids, on="barrid", how="inner")
        universe = universe.join(bsrf_ids, on="barrid", how="inner")

        # Sort and return
        self._universe = universe.sort(by="barrid")["barrid"].to_list()

    def _covariance_matrix(self) -> pl.DataFrame:
        # Load
        bfc = BarraFactorCovariances().load(self._date.year, self._date)

        # Pivot
        bfc = bfc.pivot(on="factor_2", index="factor_1", values="covariance")

        # Sort headers and columns
        bfc = bfc.select(["factor_1"] + sorted([col for col in bfc.columns if col != "factor_1"]))
        bfc = bfc.sort(by="factor_1")

        # Record factor ids
        factors = bfc.select("factor_1").to_numpy().flatten()

        # Convert from upper triangular to symetric
        utm = bfc.drop("factor_1").to_numpy()
        cov_mat = np.where(np.isnan(utm), utm.T, utm)

        # Package
        cov_mat = pl.DataFrame(
            {
                "factor_1": factors,
                **{col: cov_mat[:, idx] for idx, col in enumerate(factors)},
            }
        )

        return cov_mat

    def _exposures_matrix(self) -> pl.DataFrame:
        # Load
        bfe = BarraFactorExposures().load(self._date.year, self._date)

        # Filter
        bfe = bfe.filter(pl.col("barrid").is_in(self._universe))

        # Pivot
        exp_mat = bfe.pivot(on="factor", index="barrid", values="exposure")

        # Sort headers and rows
        exp_mat = exp_mat.select(["barrid"] + sorted([col for col in exp_mat.columns if col != "barrid"]))
        exp_mat = exp_mat.sort(by="barrid")

        # Fill null values
        exp_mat = exp_mat.fill_null(0)

        return exp_mat

    def _idio_risk_matrix(self) -> pl.DataFrame:
        # Load
        bsrf = BarraSpecificRiskForecast().load(self._date.year, self._date)

        # Filter
        bsrf = bsrf.filter(pl.col("barrid").is_in(self._universe))

        # Convert vector to diagonal matrix
        diagonal = np.power(np.diag(bsrf["specificrisk"]), 2)

        # Package
        risk_matrix = pl.DataFrame(
            {
                "barrid": self._universe,
                **{id: diagonal[:, i] for i, id in enumerate(self._universe)},
            }
        )

        return risk_matrix

    def load(self) -> pl.DataFrame:
        # Cast to numpy matrices
        exposures_matrix = self._exposures_matrix().drop("barrid").to_numpy()
        covariance_matrix = self._covariance_matrix().drop("factor_1").to_numpy()
        idio_risk_matrix = self._idio_risk_matrix().drop("barrid").to_numpy()

        # Compute risk model
        risk_model = exposures_matrix @ covariance_matrix @ exposures_matrix.T + idio_risk_matrix

        # Package
        risk_model = pl.DataFrame(
            {
                "barrid": self._universe,
                **{id: risk_model[:, i] for i, id in enumerate(self._universe)},
            }
        )

        return risk_model


if __name__ == "__main__":
    date_ = date(2024, 1, 2)
    nrm = NewRiskModel(date_=date_).load()
    print(nrm)
