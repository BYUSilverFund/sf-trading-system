import cvxpy as cp


def full_investment(weights: cp.Variable) -> cp.Constraint:
    return weights >= 0
