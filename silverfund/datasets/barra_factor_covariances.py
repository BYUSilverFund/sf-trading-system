import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class BarraFactorCovariances:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        user = os.getenv("ROOT").split("/")[2]
        root_dir = Path(f"/home/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"
        self._files = os.listdir(self._folder)

    def load(self, year: int) -> pl.DataFrame:

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

        # Split Combined colum into Barrid and Factor
        df = (
            df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
            .with_columns(
                pl.col("parts").list.first().alias("Factor1"),
                pl.col("parts").list.last().alias("Factor2"),
            )
            .drop(["Combined", "parts"])
        )

        # Melt date headers into a column
        df = df.unpivot(index=["Factor1", "Factor2"], variable_name="Date", value_name="Covariance")

        # Pivot out factor 2
        df = df.pivot(on="Factor2", index=["Date", "Factor1"])

        # Cast date type
        df = df.with_columns(pl.col("Date").str.strptime(pl.Date).dt.date())

        # # Sort
        df = df.sort(by=["Date", "Factor1"])

        return df
