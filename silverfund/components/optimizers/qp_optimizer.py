import cvxpy as cp
import numpy as np
import polars as pl


def qp(alphas: np.ndarray, covariance_matrix: np.ndarray, gamma: float = 2.0):
    """
    The assets dataframe has a barrid and alpha column
    """

    # Declare variables
    n_assets = len(alphas)
    weights = cp.Variable(n_assets)

    # Objective function
    portfolio_alpha = weights.T @ alphas
    portfolio_variance = weights.T @ covariance_matrix @ weights
    objective = cp.Maximize(portfolio_alpha - 0.5 * gamma * portfolio_variance)

    # Constraints
    constraints = []

    # Formulate problem
    problem = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    problem.solve()

    # Scale weights
    scaled_weights = weights.value / sum(weights.value)

    return scaled_weights


if __name__ == "__main__":
    # Test data setup
    assets_data = {"barrid": ["A", "B", "C"], "alpha": [0.1, 0.2, 0.15]}
    assets = pl.DataFrame(assets_data)

    covariance_matrix_data = np.array([[0.1, 0.02, 0.03], [0.02, 0.1, 0.04], [0.03, 0.04, 0.1]])

    gamma = 2.0

    # Run the function
    weights = qp(assets, covariance_matrix_data, gamma)

    # Display the result
    print("Optimized Weights:", weights)
