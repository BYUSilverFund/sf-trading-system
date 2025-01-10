from typing import Self

import polars as pl


class ChunkedData:
    def __init__(self, data: pl.DataFrame, window: int, columns: list[str]):
        self.window = window

        unique_dates = data.select("date").unique().sort(by="date")["date"].to_list()
        chunks = []

        for i in range(window, len(unique_dates) + 1):

            start_date = unique_dates[i - window]
            end_date = unique_dates[i - 1]

            chunk = data.filter(
                (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
            ).select(columns)

            chunks.append(chunk)

        self._chunks: list[pl.DataFrame] = chunks

    def apply_signal_transform(self, signal) -> Self:
        for i, chunk in enumerate(self.chunks):
            self.chunks[i] = signal(chunk)

        return self

    def apply_portfolio_gen(self, portfolio_generator) -> list[pl.DataFrame]:
        portfolios = []
        for chunk in self.chunks:
            portfolios.append(portfolio_generator(chunk))
        return portfolios

    def remove_chunks(self):
        """Remove chunks that do not have the full window of data."""
        self._chunks = [chunk for chunk in self._chunks if len(chunk.drop_nulls()) > 10]

    @property
    def chunks(self) -> list[pl.DataFrame]:
        return self._chunks
