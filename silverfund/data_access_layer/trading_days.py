import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.enums import Interval


def load_trading_days(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:

    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # Get file paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")

    daily_files_folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
    monthly_file = root_dir / "groups" / "grp_quant" / "data" / "msf.parquet"

    # Load daily data
    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Trading Days"):

        file = f"dsf_{year}.parquet"
        dfs.append(pl.read_parquet(daily_files_folder / file, columns=["date"]))

    df = pl.concat(dfs)

    # Clean daily data
    df = clean(df)

    # Add daily lags
    daily_df = df.with_columns(
        pl.col("date").shift(1).alias("date_lag_1d"),
        pl.col("date").shift(2).alias("date_lag_2d"),
    )

    if interval == Interval.MONTHLY:
        # Load
        monthly_df = pl.read_parquet(monthly_file, columns=["date"])

        # Clean
        monthly_df = clean(monthly_df)

        # Merge daily lags
        monthly_df = monthly_df.join(daily_df, on="date", how="left")

        # Add monthly lags
        monthly_df = monthly_df.with_columns(
            pl.col("date").shift(1).alias("date_lag_1m"),
            pl.col("date").shift(2).alias("date_lag_2m"),
        )

        return filter(monthly_df, start_date, end_date)

    return filter(daily_df, start_date, end_date)


def clean(df: pl.DataFrame) -> pl.DataFrame:
    # Cast date type
    df = df.with_columns(pl.col("date").dt.date())

    # Keep unique and sort
    df = df.unique().sort("date")

    return df


def filter(df: pl.DataFrame, start_date: date, end_date: date) -> pl.DataFrame:
    return df.filter(pl.col("date").is_between(start_date, end_date))
