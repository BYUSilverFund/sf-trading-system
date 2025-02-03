from abc import ABC

import cvxpy as cp
import numpy as np


class Constraint(ABC):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        pass


class FullInvestment(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        return [cp.sum(weights) == 1]


class ZeroCost(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        return [cp.sum(weights) == 0]


class UnitBeta(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        predicted_betas: np.array = kwargs.get("betas")
        return [cp.sum(cp.multiply(weights, predicted_betas)) == 1]


class LongOnly(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        return [weights >= 0]


class NoBuyingOnMargin(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        return [weights <= 1]


class ShortingLimit(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        return [weights >= -1]


class LeverageTarget(Constraint):

    @staticmethod
    def construct(**kwargs) -> list[cp.Constraint]:
        weights: cp.Variable = kwargs.get("weights")
        bmk_weights: np.array = kwargs.get("bmk_weights")
        active_leverage_target: float = kwargs.get("active_leverage_target")
        return [cp.sum(weights - bmk_weights) == active_leverage_target]
