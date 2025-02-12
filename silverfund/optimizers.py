import cvxpy as cp
import polars as pl

from silverfund.alphas import Alpha
from silverfund.covariance_matrix import CovarianceMatrix


class Portfolio(pl.DataFrame):
    pass


def quadratic_program(
    alpha: Alpha, cov_mat: CovarianceMatrix, constraints: list[callable], gamma: float = 2.0
) -> Portfolio:
    alphas = alpha.to_vector()
    cov_mat = cov_mat.to_matrix()

    # Declare variables
    n_assets = len(alphas)
    weights = cp.Variable(n_assets)

    # Objective function
    portfolio_alpha = weights.T @ alphas
    portfolio_variance = weights.T @ cov_mat @ weights
    objective = cp.Maximize(portfolio_alpha - 0.5 * gamma * portfolio_variance)

    # Constraints
    constraints = [weights >= 0]

    # Formulate problem
    problem = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    problem.solve()

    return Portfolio(
        alpha.with_columns(pl.Series(weights.value).alias("weight")).select(
            ["date", "barrid", "weight"]
        )
    )
