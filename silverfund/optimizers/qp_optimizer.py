import cvxpy as cp
import numpy as np
import polars as pl

from silverfund.optimizers.new_constraints import Constraint


def qp(
    chunk: pl.DataFrame,
    covariance_matrix: np.ndarray,
    constraints: list[Constraint],
    gamma: float = 2.0,
):
    alphas = chunk["alpha"].to_numpy()
    betas = chunk["predbeta_lag"].to_numpy()

    # Declare variables
    n_assets = len(alphas)
    weights = cp.Variable(n_assets)

    # Objective function
    portfolio_alpha = weights.T @ alphas
    portfolio_variance = weights.T @ covariance_matrix @ weights
    objective = cp.Maximize(portfolio_alpha - 0.5 * gamma * portfolio_variance)

    # Constraints
    constraints = [
        c for constraint in constraints for c in constraint.construct(weights=weights, betas=betas)
    ]

    # Formulate problem
    problem = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    problem.solve()

    # Package portfolio
    portfolio = chunk.with_columns(pl.Series(weights.value).alias("weight")).select(
        ["date", "barrid", "weight"]
    )

    return portfolio
