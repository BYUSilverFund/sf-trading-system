import polars as pl

from silverfund.enums import Interval, Weighting
from silverfund.optimizers import decile_portfolio
from silverfund.strategies.strategy import Strategy


class ReversalStrategy(Strategy):
    def __init__(self, interval: Interval):
        self._interval = interval
        self._window = {Interval.DAILY: 23, Interval.MONTHLY: 2}[self._interval]
        self._rolling_window = {Interval.DAILY: 22, Interval.MONTHLY: 1}[self._interval]
        self._skip = {Interval.DAILY: 1, Interval.MONTHLY: 1}[self._interval]

    def compute_portfolio(self, chunk: pl.DataFrame) -> list[pl.DataFrame]:

        # Reversal signal formation
        chunk = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
            pl.col("logret")
            .rolling_sum(
                window_size=self._rolling_window, min_periods=self._rolling_window, center=False
            )
            .over("permno")
            .alias("rev")
        )

        # Reversal lag/skip
        chunk = chunk.with_columns(pl.col("rev").shift(self._skip).over("permno"))

        # Filters

        # Price greater than 5
        chunk = chunk.with_columns(pl.col("prc").shift(1).over("permno").alias("prclag"))

        chunk = chunk.filter(pl.col("prclag") > 5)

        # Non-null reversal
        chunk = chunk.drop_nulls(subset="rev")

        # If there isn't enough data return an empty portfolio
        if len(chunk) < 10:
            return pl.DataFrame()

        # Generate portfolios
        portfolios = decile_portfolio(chunk, "rev", Weighting.EQUAL)

        # Long negative reversal, short positive reversal
        long_short_portfolio = pl.concat(
            [portfolios[9].with_columns(pl.col("weight") * -1), portfolios[0]]
        ).drop(["bin", "rev"])

        return long_short_portfolio

    @property
    def window(self) -> int:
        return self._window
