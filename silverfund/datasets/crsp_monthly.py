import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class CRSPMonthly:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._master_file = root_dir / "groups" / "grp_quant" / "data" / "msf.parquet"

    def load_all(self) -> pl.DataFrame:
        return self.clean(pl.read_parquet(self._master_file))

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Cast date types
        df = df.with_columns(pl.col("date").dt.date())

        # Sort
        df = df.sort(by=["permno", "date"])

        return df
