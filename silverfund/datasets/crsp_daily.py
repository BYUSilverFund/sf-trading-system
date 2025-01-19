import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class CRSPDaily:

    def __init__(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> None:
        self._start_date = start_date
        self._end_date = end_date or date.today()

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
        self._master_file = root_dir / "groups" / "grp_quant" / "data" / "dsf.parquet"
        self._files = os.listdir(self._folder)

    def load(self, year: int) -> pl.DataFrame:
        file = f"dsf_{year}.parquet"
        return self.clean(pl.read_parquet(self._folder / file))

    def load_all(self) -> pl.DataFrame:
        df = self.clean(pl.read_parquet(self._master_file))

        df = df.filter(
            pl.col("date").is_between(self._start_date, self._end_date),
            pl.col("shrcd").is_between(10, 11, closed="both"),  # Stocks
            pl.col("exchcd").is_between(1, 3, closed="both"),  # NYSE, NASDAQ, AMEX
        )

        return df

    def get_all_years(self) -> list[int]:
        years = []
        for file in self._files:
            year = file.split("_")[1].split(".")[0]
            years.append(year)

        return years

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Cast date types
        df = df.with_columns(pl.col("date").dt.date())

        # Sort
        df = df.sort(by=["permno", "date"])

        return df
