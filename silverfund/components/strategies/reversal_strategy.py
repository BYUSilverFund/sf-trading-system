import polars as pl

from silverfund.components.enums import Interval, Weighting
from silverfund.components.optimizers import decile_portfolio
from silverfund.components.strategies.strategy import Strategy


class ReversalStrategy(Strategy):
    def __init__(self, interval: Interval) -> None:
        self._interval = interval
        self._window = {
            Interval.MONTHLY: 2,
            Interval.WEEKLY: 24,
            Interval.DAILY: 24,
        }[self._interval]

    def compute_portfolio(self, chunk: pl.DataFrame) -> list[pl.DataFrame]:

        # Signal transformation
        chunk = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
            pl.col("logret")
            .rolling_sum(window_size=self._window - 1, min_periods=self._window - 1, center=False)
            .shift(1)  # Lag signal
            .over("ticker")
            .alias("rev")
        )

        chunk = chunk.drop_nulls()

        if len(chunk) < 10:
            return pl.DataFrame()

        portfolios = decile_portfolio(chunk, "rev", Weighting.EQUAL)

        # Long poor reversal, short good reversal
        long_short_portfolio = pl.concat(
            [portfolios[0], portfolios[9].with_columns(pl.col("weight") * -1)]
        ).drop(["bin", "rev"])

        return long_short_portfolio

    @property
    def window(self) -> int:
        return self._window
