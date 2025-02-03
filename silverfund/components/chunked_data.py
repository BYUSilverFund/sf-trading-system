from datetime import date

import polars as pl
from tqdm import tqdm

from silverfund.components.enums import Interval
from silverfund.components.optimizers.new_constraints import Constraint
from silverfund.components.strategies.strategy import Strategy
from silverfund.datasets.trading_days import TradingDays


class ChunkedData:
    def __init__(self, data: pl.DataFrame, interval: Interval, window: int):
        min_date = data["date"].min()
        max_date = data["date"].max()

        trading_days = (
            TradingDays(start_date=min_date, end_date=max_date, interval=interval)
            .load_all()["date"]
            .to_list()
        )

        chunks = []
        for i in tqdm(range(window, len(trading_days) + 1), desc="Chunking data"):

            start_date: date = trading_days[i - window]

            if interval == Interval.MONTHLY:
                start_date = start_date.replace(day=1)

            end_date: date = trading_days[i - 1]

            chunk = data.filter(pl.col("date").is_between(start_date, end_date))
            chunks.append(chunk)

        self._chunks: list[pl.DataFrame] = chunks

    def apply_strategy(
        self, strategy: Strategy, constraints: list[Constraint]
    ) -> list[pl.DataFrame]:
        portfolios_list = []
        for chunk in tqdm(self._chunks, desc="Running strategy"):
            portfolios = strategy.compute_portfolio(chunk, constraints)
            if not portfolios.is_empty():
                portfolios_list.append(portfolios)
        return portfolios_list

    @property
    def chunks(self) -> list[pl.DataFrame]:
        return self._chunks
