import exchange_calendars as ecals
import polars as pl
from tqdm import tqdm

from silverfund.components.enums import Interval
from silverfund.components.strategies.strategy import Strategy


class ChunkedData:
    def __init__(self, data: pl.DataFrame, interval: Interval, window: int):
        min_date = data["date"].min()
        max_date = data["date"].max()

        nyse = ecals.get_calendar("XNYS")
        schedule = nyse.sessions_in_range(min_date, max_date).to_list()
        schedule = (
            pl.DataFrame(schedule)
            .rename({"column_0": "date"})
            .with_columns(pl.col("date").dt.date())
        )

        if interval == Interval.MONTHLY:
            schedule = schedule.with_columns(pl.col("date").dt.truncate("1mo")).unique()

        schedule = (
            schedule.filter(pl.col("date") >= min_date, pl.col("date") <= max_date)
            .sort(by="date")["date"]
            .to_list()
        )

        chunks = []

        for i in tqdm(range(window, len(schedule) + 1), desc="Chunking data"):

            start_date = schedule[i - window]
            end_date = schedule[i - 1]

            chunk = data.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

            chunks.append(chunk)

        self._chunks: list[pl.DataFrame] = chunks

    def apply_strategy(self, strategy: Strategy) -> list[pl.DataFrame]:
        portfolios_list = []
        for chunk in tqdm(self._chunks, desc="Running strategy"):
            portfolios = strategy.compute_portfolio(chunk)
            if not portfolios.is_empty():
                portfolios_list.append(portfolios)
        return portfolios_list

    @property
    def chunks(self) -> list[pl.DataFrame]:
        return self._chunks
