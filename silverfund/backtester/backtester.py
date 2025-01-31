from datetime import date

import polars as pl

from silverfund.components.chunked_data import ChunkedData
from silverfund.components.enums import Interval
from silverfund.components.strategies.strategy import Strategy


class Backtester:

    def __init__(
        self,
        start_date: date,
        end_date: date,
        interval: Interval,
        historical_data: pl.DataFrame,
        strategy: Strategy,
        security_identifier: str,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.historical_data = historical_data
        self.strategy = strategy(interval)
        self._security_identifier = security_identifier

    def run(self):

        # Create chunks
        chunked_data = ChunkedData(
            data=self.historical_data,
            interval=self.interval,
            window=self.strategy.window,
        )

        # Apply strategy
        portfolios = chunked_data.apply_strategy(self.strategy)

        # Concatenate portfolios
        portfolios = pl.concat(portfolios)

        # Join historical data and portfolios
        merged = self.historical_data.join(
            portfolios, how="inner", on=["date", self._security_identifier]
        )

        # Calculate weighted returns
        merged = merged.with_columns(
            (pl.col("weight") * pl.col("ret")).alias("weighted_ret"),
        )

        # Calculte portfolio pnl
        pnl = (
            merged.group_by("date")
            .agg(
                n_assets=pl.col("date").count(),
                portfolio_ret=pl.col("weighted_ret").sum(),
            )
            .sort(by=["date"])
        )

        # Calculate portfolio cummulative returns
        pnl = pnl.with_columns(pl.col("portfolio_ret").log1p().alias("portfolio_logret"))
        pnl = pnl.with_columns(
            ((pl.col("portfolio_ret") + 1).cum_prod() - 1).alias("cumprod") * 100,
            (pl.col("portfolio_logret").cum_sum().alias("cumsum")) * 100,
        )

        return pnl
