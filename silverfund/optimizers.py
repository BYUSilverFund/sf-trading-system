import cvxpy as cp
import polars as pl

from silverfund.alphas import Alpha
from silverfund.covariance_matrix import CovarianceMatrix


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
