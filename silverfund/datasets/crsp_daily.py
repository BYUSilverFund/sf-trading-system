import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class CRSPDaily:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        user = os.getenv("ROOT").split("/")[2]
        root_dir = Path(f"/home/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
        self._master_file = root_dir / "groups" / "grp_quant" / "data" / "dsf.parquet"
        self._files = os.listdir(self._folder)

    def load(self, year: int) -> pl.DataFrame:
        file = f"dsf_{year}.parquet"
        return self.clean(pl.read_parquet(self._folder / file))

    def load_all(self) -> pl.DataFrame:
        return self.clean(pl.read_parquet(self._master_file))

    def get_all_years(self) -> list[int]:
        years = []
        for file in self._files:
            year = file.split("_")[1].split(".")[0]
            years.append(year)

        return years

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Cast date types
        df = df.with_columns(pl.col("date").dt.date())

        # Sort
        df = df.sort(by=["permno", "date"])

        return df
