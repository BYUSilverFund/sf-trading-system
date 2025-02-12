from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import polars as pl

from silverfund.alphas import Alpha
from silverfund.covariance_matrix import CovarianceMatrix
from silverfund.optimizers import Portfolio
from silverfund.scores import Score
from silverfund.signals import Signal


@dataclass
class Strategy:
    signal_fn: Callable[[pl.DataFrame], Signal]
    score_fn: Callable[[Signal], Score]
    alpha_fn: Callable[[Score], Alpha]
    optimizer: Callable[[Alpha, CovarianceMatrix, list[cp.Variable], float], Portfolio]
    constraints: list[cp.Constraint]
