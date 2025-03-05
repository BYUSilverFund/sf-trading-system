import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm


def load_mega_merge(
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Loads Mega Merge daily data for a specified time interval.

    This function reads daily return data from Parquet files stored in a specific
    directory, cleans the data, and optionally aggregates it to a monthly frequency.
    The data is filtered to the given date range and sorted before returning.

    Args:
        start_date (date, optional): The start date for filtering the data. Defaults
            to July 31, 1995, if not provided.
        end_date (date, optional): The end date for filtering the data. Defaults to
            the current date if not provided.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the filtered and processed
        Barra returns data.

    Example:
        >>> from datetime import date
        >>> df = load_mega_merge(start_date=date(2020, 1, 1), end_date=date(2021, 12, 31))
        >>> print(df.head())
    """

    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # File paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "mega_merge"

    # Load daily data
    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Mega Merge"):
        file = f"mm_{year}.parquet"

        # Load
        df = pl.read_parquet(folder / file)

        # Clean
        df = clean(df)

        # Append
        dfs.append(df)

    # Concat
    df: pl.DataFrame = pl.concat(dfs)

    # Filter
    df = df.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    df = df.sort(by=["barrid", "date"])

    return df


def clean(df: pl.DataFrame):
    # Cast and rename date
    df = df.with_columns(pl.col("DataDate").dt.date().alias("date"))

    # Drop date columns
    df = df.drop(["DataDate", "obsdate", "enddate"])

    # Lowercase columns
    df = df.rename({col: col.lower() for col in df.columns})

    # Reorder columns
    df = df.select(
        ["date", "barrid"] + [col for col in sorted(df.columns) if col not in ["date", "barrid"]]
    )

    # Cast columns
    df = df.cast({"permno": pl.String})

    return df
