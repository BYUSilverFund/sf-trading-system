from datetime import date

import numpy as np
import polars as pl

import silverfund.data_access_layer as dal


class CovarianceMatrix(pl.DataFrame):
    def __init__(self, cov_mat: pl.DataFrame, barrids: list[str], date_: date) -> None:
        expected_order = ["barrid"] + sorted(barrids)

        valid_schema = {barrid: pl.Float64 for barrid in barrids}

        # Check if all required columns exist
        if set(expected_order) != set(cov_mat.columns):
            missing = set(expected_order) - set(cov_mat.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure correct column types
        for col, dtype in valid_schema.items():
            if cov_mat.schema[col] != dtype:
                raise ValueError(
                    f"Column {col} has incorrect type: {cov_mat.schema[col]}, expected: {dtype}"
                )

        # Reorder columns
        cov_mat = cov_mat.select(expected_order)

        # Initialize
        super().__init__(cov_mat)

    def to_matrix(self):
        return self.drop("barrid").to_numpy()


def covariance_matrix_constructor(date_: date, barrids: list[str]) -> CovarianceMatrix:
    # Load
    exposures_matrix = factor_exposure_matrix_constructor(date_, barrids).drop("barrid").to_numpy()
    covariance_matrix = factor_covariance_matrix_constructor(date_).drop("factor_1").to_numpy()
    idio_risk_matrix = specific_risk_matrix(date_, barrids).drop("barrid").to_numpy()

    # Compute covariance matrix
    covariance_matrix = exposures_matrix @ covariance_matrix @ exposures_matrix.T + idio_risk_matrix

    # Package
    covariance_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: covariance_matrix[:, i] for i, id in enumerate(barrids)},
        }
    )

    covariance_matrix = covariance_matrix.fill_nan(0)

    return CovarianceMatrix(covariance_matrix, barrids=barrids, date_=date_)


def factor_exposure_matrix_constructor(date_: date, barrids: list[str]) -> pl.DataFrame:
    # Load
    bfe = dal.load_factor_exposures(date_)

    # Filter
    bfe = bfe.filter(pl.col("barrid").is_in(barrids))

    # Pivot
    exp_mat = bfe.pivot(on="factor", index="barrid", values="exposure")

    # Sort headers and rows
    exp_mat = exp_mat.select(
        ["barrid"] + sorted([col for col in exp_mat.columns if col != "barrid"])
    )
    exp_mat = exp_mat.sort(by="barrid")

    # Fill null values
    exp_mat = exp_mat.fill_null(0)

    return exp_mat


def factor_covariance_matrix_constructor(date_: date) -> pl.DataFrame:
    # Load
    fc_df = dal.load_factor_covariances(date_)

    # Pivot
    fc_df = fc_df.pivot(on="factor_2", index="factor_1", values="covariance")

    # Sort headers and columns
    fc_df = fc_df.select(["factor_1"] + sorted([col for col in fc_df.columns if col != "factor_1"]))
    fc_df = fc_df.sort(by="factor_1")

    # Record factor ids
    factors = fc_df.select("factor_1").to_numpy().flatten()

    # Convert from upper triangular to symetric
    utm = fc_df.drop("factor_1").to_numpy()
    cov_mat = np.where(np.isnan(utm), utm.T, utm)

    # Package
    cov_mat = pl.DataFrame(
        {
            "factor_1": factors,
            **{col: cov_mat[:, idx] for idx, col in enumerate(factors)},
        }
    )

    # Fill NaN values
    cov_mat = cov_mat.fill_nan(0)

    return cov_mat


def specific_risk_matrix(date_: date, barrids: list[str]) -> pl.DataFrame:
    # Load
    sr_df = dal.load_specific_risk(date_)

    # Filter
    sr_df = sr_df.filter(pl.col("barrid").is_in(barrids))

    # Convert vector to diagonal matrix
    diagonal = np.power(np.diag(sr_df["specific_risk"]), 2)

    # Package
    risk_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: diagonal[:, i] for i, id in enumerate(barrids)},
        }
    )

    return risk_matrix
