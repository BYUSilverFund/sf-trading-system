from abc import ABC

import cvxpy as cp


class Constraint(ABC):

    @staticmethod
    def construct(weights: cp.Variable) -> list[cp.Constraint]:
        pass


class FullInvestment(Constraint):

    @staticmethod
    def construct(weights: cp.Variable) -> list[cp.Constraint]:
        return [cp.sum(weights) == 1]


class NoLeverage(Constraint):

    @staticmethod
    def construct(weights: cp.Variable) -> list[cp.Constraint]:
        return [weights >= -1, weights <= 1]


class ZeroCost(Constraint):

    @staticmethod
    def construct(weights: cp.Variable) -> list[cp.Constraint]:
        return [cp.sum(weights) == 0]
