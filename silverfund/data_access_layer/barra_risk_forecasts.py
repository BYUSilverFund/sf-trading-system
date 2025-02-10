import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm


def load_risk_forecasts(
    start_date: date,
    end_date: date,
):
    # Parameters
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow_asset"

    # Load daily data
    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Barra Risk Forecasts"):
        file = f"asset_{year}.parquet"

        # Load
        df = pl.read_parquet(folder / file)

        # Clean
        df = clean(df)

        # Append
        dfs.append(df)

    # Concat
    dfs = pl.concat(dfs)

    # Filter
    dfs = dfs.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    df = df.sort(by=["date", "barrid"])

    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
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
