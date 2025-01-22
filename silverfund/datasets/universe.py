import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv

from silverfund.datasets.exchange_calendar import ExchangeCalendar


class Universe:

    def __init__(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        self._start_date = start_date or date(1995, 7, 31)  # min date in russell history file
        self._end_date = end_date or date.today()

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._file_path = root_dir / "groups" / "grp_quant" / "data" / "russell_history.parquet"

    def _calendar(self):
        return ExchangeCalendar().load()

    def _barrids(self):
        df = pl.read_parquet(self._file_path)

        # Cast types
        df = df.with_columns(pl.col("date").dt.date())

        # Drop null barrids
        df = df.drop_nulls(subset=["barrid"])

        # Select
        df = df.select(["date", "barrid"]).unique().sort(by=["date", "barrid"])

        # Add in_universe flag
        df = df.with_columns(pl.lit(True).alias("in_universe"))

        # Pivot and fill
        df = df.pivot(on="barrid", index="date", values="in_universe").fill_null(False)

        return df

    def _merge(self) -> pl.DataFrame:
        # Load
        calendar = self._calendar()
        barrids = self._barrids()

        # Merge
        merge = calendar.join(barrids, on="date", how="left")

        # Forward fill
        merge = merge.fill_null(strategy="forward")

        # Unpivot
        merge = merge.unpivot(index="date", variable_name="barrid", value_name="in_universe")

        # Drop nulls
        merge = merge.drop_nulls(subset="in_universe")

        return merge

    def load(self) -> pl.DataFrame:
        df = self._merge()

        # Filter
        df = df.filter(pl.col("date").is_between(self._start_date, self._end_date), pl.col("in_universe"))

        # Drop in_universe column
        df = df.drop("in_universe")

        return df


if __name__ == "__main__":
    univ = Universe(start_date=date(1995, 1, 1), end_date=date(2024, 12, 31)).load()

    print(univ)
