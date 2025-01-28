import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.components.enums import Interval
from silverfund.datasets.crsp_daily import CRSPDaily
from silverfund.datasets.crsp_monthly import CRSPMonthly


class TradingDays:

    def __init__(self, start_date: date, end_date: date, interval: Interval) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
        self._master_file = root_dir / "groups" / "grp_quant" / "data" / "msf.parquet"
        self._files = os.listdir(self._folder)

    def load_all(self) -> pl.DataFrame:

        if self._interval == Interval.DAILY:
            # Join all yearly files
            years = range(self._start_date.year, self._end_date.year + 1)
            dfs = []
            for year in tqdm(years, desc="Loading daily data"):
                file = f"dsf_{year}.parquet"
                dfs.append(pl.read_parquet(self._folder / file, columns=["date"]))

            df = pl.concat(dfs)

            return self.clean(df)

        elif self._interval == Interval.MONTHLY:
            df = pl.read_parquet(self._master_file, columns=["date"])
            return self.clean(df)

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        # Cast date type
        df = df.with_columns(pl.col("date").dt.date())

        # Filter
        df = df.filter(pl.col("date").is_between(self._start_date, self._end_date))

        # Keep unique and sort
        df = df.unique().sort("date")

        return df
