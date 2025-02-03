import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv


class BarraFactorCovariances:

    def __init__(self) -> None:
        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"
        self._files = os.listdir(self._folder)

    def load(self, year: int, date: Optional[date] = None) -> pl.DataFrame:

        file = f"factor_covariance_{year}.parquet"

        date_column = date.strftime("%Y-%m-%d 00:00:00") if date else None
        columns = ["Combined", date_column] if date else None

        # Optionally read just a specific days covariances
        df = pl.read_parquet(self._folder / file, columns=columns)

        # Rename date column
        df = df.rename({date_column: "covariance"}) if date_column else df

        # Split Combined column into factor_1 and factor_2
        df = (
            df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
            .with_columns(
                pl.col("parts").list.first().alias("factor_1"),
                pl.col("parts").list.last().alias("factor_2"),
            )
            .drop(["Combined", "parts"])
        )

        # Reorder columns
        columns = ["factor_1", "factor_2"] + [
            col for col in df.columns if col not in ["factor_1", "factor_2"]
        ]

        df = df.select(columns)

        return df

    def load_pivoted(self, year: int) -> pl.DataFrame:

        file = f"factor_covariance_{year}.parquet"

        return self.clean(pl.read_parquet(self._folder / file))

    def get_all_years(self) -> list[int]:

        years = []
        for file in self._files:
            file_arr = file.split("_")
            if file_arr[0] == "factor":
                year = file_arr[2].split(".")[0]
                years.append(year)

        return years

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Rename columns headers (cast to date)
        new_cols = list(map(lambda x: x.split(" ")[0], df.columns))
        df = df.rename({col: new_col for col, new_col in zip(df.columns, new_cols)})

        # Split Combined column into Barrid and Factor
        df = (
            df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
            .with_columns(
                pl.col("parts").list.first().alias("factor_1"),
                pl.col("parts").list.last().alias("factor_2"),
            )
            .drop(["Combined", "parts"])
        )

        # Melt date headers into a column
        df = df.unpivot(
            index=["factor_1", "factor_2"], variable_name="date", value_name="covariance"
        )

        # Pivot out factor 2
        df = df.pivot(on="factor_2", index=["date", "factor_1"])

        # Cast date type
        df = df.with_columns(pl.col("date").str.strptime(pl.Date).dt.date())

        # # Sort
        df = df.sort(by=["date", "factor_1"])

        return df
