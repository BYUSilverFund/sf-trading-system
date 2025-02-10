import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.datasets.crsp_daily import CRSPDaily
from silverfund.datasets.crsp_monthly import CRSPMonthly
from silverfund.enums import Interval


class TradingDays:

    def __init__(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: Interval = Interval.MONTHLY,
        quiet: bool = True,
    ) -> None:
        self._start_date = start_date or date(1995, 7, 31)  # min date in russell history file
        self._end_date = end_date or date.today()
        self._interval = interval
        self._quiet = quiet

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
        self._master_file = root_dir / "groups" / "grp_quant" / "data" / "msf.parquet"
        self._files = os.listdir(self._folder)

    def load_all(self) -> pl.DataFrame:
        # Join all yearly files
        years = range(self._start_date.year, self._end_date.year + 1)
        dfs = []

        if not self._quiet:
            for year in tqdm(years, desc="Loading daily data"):
                file = f"dsf_{year}.parquet"
                dfs.append(pl.read_parquet(self._folder / file, columns=["date"]))
        else:
            for year in years:
                file = f"dsf_{year}.parquet"
                dfs.append(pl.read_parquet(self._folder / file, columns=["date"]))

        df = pl.concat(dfs)

        return self.transform(df)

    def _clean(self, df: pl.DataFrame) -> pl.DataFrame:
        # Cast date type
        df = df.with_columns(pl.col("date").dt.date())

        # Filter
        df = df.filter(pl.col("date").is_between(self._start_date, self._end_date))

        # Keep unique and sort
        df = df.unique().sort("date")

        return df

    def transform(self, daily_df: pl.DataFrame) -> pl.DataFrame:
        daily_df = self._clean(daily_df)

        # Add daily lags
        daily_df = daily_df.with_columns(
            pl.col("date").shift(1).alias("date_lag_1d"),
            pl.col("date").shift(2).alias("date_lag_2d"),
        )

        if self._interval == Interval.MONTHLY:
            # Load
            monthly_df = pl.read_parquet(self._master_file, columns=["date"])

            # Clean
            monthly_df = self._clean(monthly_df)

            # Merge daily lags
            monthly_df = monthly_df.join(daily_df, on="date", how="left")

            # Add monthly lags
            monthly_df = monthly_df.with_columns(
                pl.col("date").shift(1).alias("date_lag_1m"),
                pl.col("date").shift(2).alias("date_lag_2m"),
            )

            return monthly_df

        return daily_df
