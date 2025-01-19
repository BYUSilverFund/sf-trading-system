import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class BarraFactorExposures:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"
        self._files = os.listdir(self._folder)

    def load(self, year: int, date: Optional[date] = None) -> pl.DataFrame:

        file = f"exposures_{year}.parquet"

        date_column = date.strftime("%Y-%m-%d 00:00:00") if date else None
        columns = ["Combined", date_column] if date else None

        # Optionally read just a specific days exposures
        df = pl.read_parquet(self._folder / file, columns=columns)

        # Rename date column
        df = df.rename({date_column: "exposure"}) if date_column else df

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
        columns = ["barrid", "factor"] + [
            col for col in df.columns if col not in ["barrid", "factor"]
        ]
        df = df.select(columns)

        return df

    def load_pivoted(self, year: int) -> pl.DataFrame:

        file = f"exposures_{year}.parquet"

        return self.clean(pl.read_parquet(self._folder / file))

    def get_all_years(self) -> list[int]:

        years = []
        for file in self._files:
            file_arr = file.split("_")
            if file_arr[0] == "exposures":
                year = file_arr[1].split(".")[0]
                years.append(year)

        return years

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Rename columns headers (cast to date)
        new_cols = list(map(lambda x: x.split(" ")[0], df.columns))
        df = df.rename({col: new_col for col, new_col in zip(df.columns, new_cols)})

        # Split Combined colum into Barrid and Factor
        df = (
            df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
            .with_columns(
                pl.col("parts").list.first().alias("barrid"),
                pl.col("parts").list.last().alias("factor"),
            )
            .drop(["Combined", "parts"])
        )

        # Melt date headers into a column
        df = df.unpivot(index=["barrid", "factor"], variable_name="date", value_name="exposure")

        # Cast date type
        df = df.with_columns(pl.col("date").str.strptime(pl.Date).dt.date())

        # Sort
        df = df.sort(by=["barrid", "date"])

        return df
