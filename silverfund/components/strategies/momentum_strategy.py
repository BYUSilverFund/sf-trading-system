import polars as pl

from silverfund.components.enums import Interval, Weighting
from silverfund.components.optimizers import decile_portfolio
from silverfund.components.strategies.strategy import Strategy


class MomentumStrategy(Strategy):
    def __init__(self, interval: Interval):
        self._interval = interval
        self._window = {Interval.DAILY: 252, Interval.MONTHLY: 13}[self._interval]
        self._rolling_window = {Interval.DAILY: 230, Interval.MONTHLY: 11}[self._interval]
        self._lag = {Interval.DAILY: 22, Interval.MONTHLY: 2}[self._interval]

    def compute_portfolio(self, chunk: pl.DataFrame) -> list[pl.DataFrame]:

        # Momentum signal formation
        chunk = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
            pl.col("logret").rolling_sum(window_size=self._rolling_window, min_periods=self._rolling_window, center=False).over("permno").alias("mom")
        )

        # Momentum lag/skip
        chunk = chunk.with_columns(pl.col("mom").shift(self._lag).over("permno"))

        # Filters

        # Price greater than 5
        chunk = chunk.with_columns(pl.col("prc").shift(1).over("permno").alias("prclag"))

        chunk = chunk.filter(pl.col("prclag") > 5)

        # Non-null momentum
        chunk = chunk.drop_nulls(subset="mom")

        # If there isn't enough data return an empty portfolio
        if len(chunk) < 10:
            return pl.DataFrame()

        # Generate portfolios
        portfolios = decile_portfolio(chunk, "mom", Weighting.EQUAL)

        # Long good momentum, short poor momentum
        long_short_portfolio = pl.concat([portfolios[0].with_columns(pl.col("weight") * -1), portfolios[9]]).drop(["bin", "mom"])

        return long_short_portfolio

    @property
    def window(self) -> int:
        return self._window
