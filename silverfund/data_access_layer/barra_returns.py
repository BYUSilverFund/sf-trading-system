import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.enums import Interval


def load_barra_returns(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # File paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow_ret"

    # Load daily data
    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Barra Returns"):
        file = f"ret_{year}.parquet"

        # Load
        df = pl.read_parquet(folder / file)

        # Clean
        df = clean(df)

        # Append
        dfs.append(df)

    # Concat
    df = pl.concat(dfs)

    # Aggregate to monthly
    df = aggregate_to_monthly(df) if interval == Interval.MONTHLY else df

    # Filter
    df = df.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    df = df.sort(by=["barrid", "date"])

    return df


def clean(df: pl.DataFrame):
    # Drop index column
    df = df.drop("__index_level_0__")

    # Cast and rename date
    df = df.with_columns(pl.col("DataDate").dt.date().alias("date")).drop("DataDate")

    # Lowercase columns
    df = df.rename({col: col.lower() for col in df.columns})

    # Reorder columns
    df = df.select(
        ["date", "barrid"] + [col for col in sorted(df.columns) if col not in ["date", "barrid"]]
    )

    return df


def aggregate_to_monthly(df: pl.DataFrame) -> pl.DataFrame:
    # Add logret column
    df = df.with_columns(pl.col("ret").log1p().alias("logret"))

    # Add month column
    df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).sort(["barrid", "date"])

    df = df.group_by(["month", "barrid"]).agg(
        pl.col("date").last(),
        pl.col("currency").last(),
        pl.col("mktcap").last(),
        pl.col("price").last(),
        pl.col("logret").sum(),
    )

    # Compound up log returns
    df = df.with_columns((pl.col("logret").exp() - 1).alias("ret"))

    # Drop month
    df = df.drop("month")

    return df
