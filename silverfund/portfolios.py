from datetime import date
from functools import partial
from typing import Protocol

import polars as pl

from silverfund.alphas import Alpha
from silverfund.constraints import ConstraintConstructor
from silverfund.covariance_matrix import covariance_matrix_constructor
from silverfund.optimizers import quadratic_program
from silverfund.records import Portfolio


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

    portfolio = pl.DataFrame({"date": period, "barrid": barrids, "weight": weights})
    portfolio = portfolio.sort(["barrid", "date"])

    return Portfolio(portfolio)
