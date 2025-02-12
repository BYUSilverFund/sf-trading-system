from datetime import date
from functools import partial
from typing import Protocol

import polars as pl

from silverfund.alphas import Alpha
from silverfund.constraints import ConstraintConstructor
from silverfund.covariance_matrix import covariance_matrix_constructor
from silverfund.optimizers import quadratic_program


class Portfolio(pl.DataFrame):
    def __init__(self, portfolios: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "weight"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "weight": pl.Float64,
        }

        # Check if all required columns exist
        if set(expected_order) != set(portfolios.columns):
            missing = set(expected_order) - set(portfolios.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure correct column types
        for col, dtype in valid_schema.items():
            if portfolios.schema[col] != dtype:
                raise ValueError(
                    f"Column {col} has incorrect type: {portfolios.schema[col]}, expected: {dtype}"
                )

        # Reorder columns
        portfolios = portfolios.select(expected_order)

        # Initialize
        super().__init__(portfolios)


class PortfolioConstructor(Protocol):
    def __call__(
        self,
        period: date,
        barrids: list[str],
        alphas: Alpha,
        constraints: list[ConstraintConstructor],
    ) -> Portfolio: ...


def mean_variance_efficient(
    period: date,
    barrids: list[str],
    alphas: Alpha,
    constraints: list[ConstraintConstructor],
    gamma: float = 2.0,
) -> Portfolio:

    # Get covariance matrix
    cov_mat = covariance_matrix_constructor(period, barrids)

    # Cast to numpy arrays
    alphas = alphas.to_vector()
    cov_mat = cov_mat.to_matrix()

    # Construct constraints
    constraints = [partial(constraint, date_=period, barrids=barrids) for constraint in constraints]

    # Find optimal weights
    weights = quadratic_program(alphas, cov_mat, constraints, gamma)

    return Portfolio(pl.DataFrame({"date": period, "barrid": barrids, "weight": weights}))
