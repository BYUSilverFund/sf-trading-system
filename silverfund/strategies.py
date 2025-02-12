from dataclasses import dataclass

from silverfund.alphas import AlphaConstructor
from silverfund.constraints import ConstraintConstructor
from silverfund.portfolios import PortfolioConstructor
from silverfund.scores import ScoreConstructor
from silverfund.signals import SignalConstructor


@dataclass
class Strategy:
    signal_constructor: SignalConstructor
    score_constructor: ScoreConstructor
    alpha_constructor: AlphaConstructor
    portfolio_constructor: PortfolioConstructor
    constraints: list[ConstraintConstructor]
