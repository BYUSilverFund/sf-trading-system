from abc import ABC

import cvxpy as cp


class Constraint(ABC):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        pass


class FullInvestment(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights = kwargs.get("weights")
        return [cp.sum(weights) == 1]


class NoLeverage(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights = kwargs.get("weights")
        return [weights >= -1, weights <= 1]


class ZeroCost(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights = kwargs.get("weights")
        return [cp.sum(weights) == 0]


class UnitBeta(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights = kwargs.get("weights")
        betas = kwargs.get("betas")
        return [cp.sum(cp.multiply(weights, betas)) == 1]


class LongOnly(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights = kwargs.get("weights")
        return [weights >= 0]
