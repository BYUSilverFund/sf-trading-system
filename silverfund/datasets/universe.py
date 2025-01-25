from datetime import date
from typing import Optional

import polars as pl

from silverfund.datasets.russell_constituents import RussellConstituents
from silverfund.datasets.trading_days import TradingDays


class Universe:

    def __init__(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        self._start_date = start_date or date(1995, 7, 31)  # min date in russell history file
        self._end_date = end_date or date.today()

    def _trading_days(self):
        trading_days = TradingDays(start_date=self._start_date, end_date=self._end_date).load()

        return trading_days

    def _russell_constituents(self):
        russell = RussellConstituents().load_all()

        russell = russell.select(["date", "barrid"])

        russell = russell.drop_nulls()

        russell = russell.with_columns(pl.lit(True).alias("in_universe"))

        russell = russell.pivot(on="barrid", index="date", values="in_universe").fill_null(False)

        return russell

    def _merge(self) -> pl.DataFrame:
        # Load
        trading_days = self._trading_days()
        russell = self._russell_constituents()

        # Merge
        merged = trading_days.join(russell, on="date", how="left")

        # Forward fill
        merged = merged.fill_null(strategy="forward").drop_nulls()

        # Unpivot
        merge = merged.unpivot(value_name="in_universe", variable_name="barrid", index="date")

        # Keep in universe
        merge = merge.filter(pl.col("in_universe"))

        # Drop in_universe column
        merge = merge.drop("in_universe")

        return merge

    def load(self) -> pl.DataFrame:
        df = self._merge()

        # Filter
        df = df.filter(pl.col("date").is_between(self._start_date, self._end_date))

        return df


if __name__ == "__main__":
    univ = Universe(start_date=date(1995, 7, 31), end_date=date(2024, 12, 31)).load()

    print(univ)
