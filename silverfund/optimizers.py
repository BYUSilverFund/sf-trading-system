from typing import Protocol

import cvxpy as cp
import numpy as np

from silverfund.constraints import ConstraintConstructor


class Optimizer(Protocol):
    def __call__(
        self,
        alphas: np.array,
        cov_mat: np.array,
        constraints: list[ConstraintConstructor],
        gamma: float = 2.0,
    ) -> np.array: ...


def quadratic_program(
    alphas: np.array, cov_mat: np.array, constraints: list[ConstraintConstructor], gamma: float
) -> np.array:

    # Declare variables
    n_assets = len(alphas)
    weights = cp.Variable(n_assets)

    constraints = [constraint(weights) for constraint in constraints]

    # Objective function
    portfolio_alpha = weights.T @ alphas
    portfolio_variance = weights.T @ cov_mat @ weights
    objective = cp.Maximize(portfolio_alpha - 0.5 * gamma * portfolio_variance)

    # Formulate problem
    problem = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    problem.solve(solver=cp.OSQP)

    return weights.value
