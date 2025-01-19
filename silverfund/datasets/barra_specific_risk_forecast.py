import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class BarraSpecificRiskForecast:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"
        self._files = os.listdir(self._folder)

    def load_raw(self, year: int) -> pl.DataFrame:

        file = f"spec_risk_{year}.parquet"

        return pl.read_parquet(self._folder / file)

    def load_clean(self, year: int) -> pl.DataFrame:

        file = f"spec_risk_{year}.parquet"

        return self.clean(pl.read_parquet(self._folder / file))

    def get_all_years(self) -> list[int]:

        years = []
        for file in self._files:
            file_arr = file.split("_")
            if file_arr[0] == "spec":
                year = file_arr[2].split(".")[0]
                years.append(year)

        return years

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Rename columns headers (cast to date)
        new_cols = list(map(lambda x: x.split(" ")[0], df.columns))
        df = df.rename({col: new_col for col, new_col in zip(df.columns, new_cols)})

        # Melt date headers into a column
        df = df.unpivot(index="Barrid", variable_name="Date", value_name="SpecificRisk")

        # Cast date type
        df = df.with_columns(pl.col("Date").str.strptime(pl.Date).dt.date())

        # Reorder columns
        df = df.select(["Date", "Barrid", "SpecificRisk"])

        # Sort
        df = df.sort(by=["Barrid", "Date"])

        return df
