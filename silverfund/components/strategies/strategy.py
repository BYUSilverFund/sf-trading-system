from abc import ABC, abstractmethod

import polars as pl

from silverfund.components.enums import Interval


class Strategy(ABC):
    @abstractmethod
    def __init__(self, interval: Interval) -> None:
        pass

    @abstractmethod
    def compute_portfolio(self, data: pl.DataFrame) -> pl.DataFrame:
        pass

    @property
    @abstractmethod
    def window(self) -> int:
        pass
