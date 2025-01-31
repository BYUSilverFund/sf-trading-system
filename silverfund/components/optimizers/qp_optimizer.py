import cvxpy as cp
import numpy as np
import polars as pl

from silverfund.components.optimizers.new_constraints import Constraint


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

    # Scale weights
    scaled_weights = weights.value / sum(weights.value)

    # Package portfolio
    portfolio = chunk.with_columns(pl.Series(scaled_weights).alias("weight")).select(
        ["date", "barrid", "weight"]
    )

    return portfolio


if __name__ == "__main__":
    # Test data setup
    alphas = np.array([0.1, 0.2, 0.15])
    covariance_matrix_data = np.array([[0.1, 0.02, 0.03], [0.02, 0.1, 0.04], [0.03, 0.04, 0.1]])

    # Run the function
    weights = qp(alphas, covariance_matrix_data)

    # Display the result
    print("Optimized Weights:", weights)
