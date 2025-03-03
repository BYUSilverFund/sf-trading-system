import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

BARRA_RISK_FACTORS = [
    "BETA",
    "DIVYILD",
    "EARNQLTY",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
    "LIQUIDTY",
    "LTREVRSL",
    "MGMTQLTY",
    "MIDCAP",
    "MOMENTUM",
    "PROFIT",
    "PROSPECT",
    "RESVOL",
    "SIZE",
    "VALUE",
]

BARRA_INDUSTRY_FACTORS = [
    "OILGSDRL",
    "OILGSEQP",
    "OILGSEXP",
    "OILGSCON",
    "CHEM",
    "SPTYCHEM",
    "CNSTMATL",
    "CONTAINR",
    "PAPER",
    "ALUMSTEL",
    "PRECMTLS",
    "AERODEF",
    "CNSTMACH",
    "INDMACH",
    "BLDGPROD",
    "TRADECO",
    "CNSTENG",
    "ELECEQP",
    "CONGLOM",
    "COMSVCS",
    "AIRLINES",
    "TRANSPRT",
    "ROADRAIL",
    "AUTO",
    "HOUSEDUR",
    "HOMEBLDG",
    "LEISPROD",
    "LEISSVCS",
    "RESTAUR",
    "MEDIA",
    "DISTRIB",
    "NETRET",
    "APPAREL",
    "SPTYSTOR",
    "SPLTYRET",
    "PSNLPROD",
    "FOODRET",
    "FOODPROD",
    "BEVTOB",
    "HLTHEQP",
    "MGDHLTH",
    "HLTHSVCS",
    "BIOLIFE",
    "PHARMA",
    "BANKS",
    "DIVFIN",
    "LIFEINS",
    "INSURNCE",
    "REALEST",
    "SEMIEQP",
    "SEMICOND",
    "INTERNET",
    "SOFTWARE",
    "COMMEQP",
    "COMPELEC",
    "TELECOM",
    "WIRELESS",
    "ELECUTIL",
    "GASUTIL",
    "MULTUTIL",
    "COUNTRY",
]


def load_factor_exposures_by_date(
    date_: date,
) -> pl.DataFrame:
    """Loads factor exposure data for a given date.

    This function retrieves factor exposures from the Barra dataset, extracting relevant information
    and splitting the combined identifier column into `barrid` and `factor`.

    Args:
        date_ (date): The date for which factor exposure data is needed.

    Returns:
        pl.DataFrame: A DataFrame containing factor exposures with columns `barrid`, `factor`, and `exposure`.

    Example:
        >>> df = load_factor_exposures(date(2023, 5, 15))
        >>> print(df)
    """

    # Paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"

    # Load
    file = f"exposures_{date_.year}.parquet"
    date_column = date_.strftime("%Y-%m-%d 00:00:00") if date else None
    columns = ["Combined", date_column]
    df = pl.read_parquet(folder / file, columns=columns)

    # Rename date column
    df = df.rename({date_column: "exposure"})

    # Split Combined colum into barrid and factor
    df = (
        df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
        .with_columns(
            pl.col("parts").list.first().alias("barrid"),
            pl.col("parts").list.last().alias("factor"),
        )
        .drop(["Combined", "parts"])
    )

    # Reorder columns
    df = df.select(["barrid", "factor", "exposure"])

    return df


def load_factor_exposures(
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Loads factor exposure data for a given date range.

    This function retrieves factor exposures from the Barra dataset, extracting relevant information
    and splitting the combined identifier column into `barrid` and `factor`.

    Args:
        start_date (date): The start date for which factor exposure data is needed.
        end_date (date): The end date for which factor exposure data is needed.

    Returns:
        pl.DataFrame: A DataFrame containing factor exposures with columns `barrid`, `factor`, and `exposure`.

    Example:
        >>> df = load_factor_exposures(date(2023, 5, 15))
        >>> print(df)
    """

    # Paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"

    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Barra factor exposures"):
        # Load
        file = f"exposures_{year}.parquet"
        df = pl.read_parquet(folder / file)
        # Clean
        df = clean(df)
        dfs.append(df)

    df: pl.DataFrame = pl.concat(dfs)

    df = df.filter(pl.col("date").is_between(start_date, end_date))

    df = df.sort(["barrid", "factor", "date"])

    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    # Split Combined column into barrid and factor
    df = (
        df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
        .with_columns(
            pl.col("parts").list.first().alias("barrid"),
            pl.col("parts").list.last().alias("factor"),
        )
        .drop(["Combined", "parts"])
    )

    # Rename date columns
    df = df.rename(
        {col: col.split(" ")[0] for col in df.columns if col not in ["barrid", "factor"]}
    )

    # Unpivot
    df = df.unpivot(index=["barrid", "factor"], variable_name="date", value_name="exposure")

    # # Pivot
    # df = df.pivot(
    #     on='factor', index=['barrid', 'date'], values='exposure'
    # )

    # Cast
    df = df.cast({"date": pl.Date})

    # df = df.select('barrid', 'date', BARRA_FACTORS)

    return df
