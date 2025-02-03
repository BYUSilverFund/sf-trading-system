from abc import ABC, abstractmethod

import polars as pl

from silverfund.enums import Interval
from silverfund.optimizers.new_constraints import Constraint


class Strategy(ABC):
    @abstractmethod
    def __init__(self, interval: Interval) -> None:
        pass

    @abstractmethod
    def compute_portfolio(self, data: pl.DataFrame, constraints: list[Constraint]) -> pl.DataFrame:
        pass

    @property
    @abstractmethod
    def window(self) -> int:
        pass
