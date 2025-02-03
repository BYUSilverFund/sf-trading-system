import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


class BarraReturns:

    def __init__(self) -> None:

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow_ret"
        self._files = os.listdir(self._folder)

    def load(self, year: int) -> pl.DataFrame:

        file = f"ret_{year}.parquet"

        return self.clean(pl.read_parquet(self._folder / file))

    def get_all_years(self) -> list[int]:

        years = []
        for file in self._files:
            year = file.split("_")[1].split(".")[0]
            years.append(year)

        return years

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:
        # Drop index column
        df = df.drop("__index_level_0__")

        # Cast and rename date
        df = df.with_columns(pl.col("DataDate").dt.date().alias("date")).drop("DataDate")

        # Lowercase columns
        df = df.rename({col: col.lower() for col in df.columns})

        # Reorder columns
        df = df.select(
            ["date", "barrid"] + [col for col in df.columns if col not in ["date", "barrid"]]
        )

        # Sort
        df = df.sort(by=["barrid", "date"])

        return df
