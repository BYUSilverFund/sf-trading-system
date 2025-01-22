import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class SecurityMapping:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._file_path = root_dir / "groups" / "grp_quant" / "data" / "usslow_ids.parquet"

    def load_all(self) -> pl.DataFrame:

        return self.clean(pl.read_parquet(self._file_path))

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Cast date types
        df = df.with_columns(
            pl.col("start_date").dt.date(),
            pl.col("end_date").dt.date(),
        )

        # Reorder columns
        dates = ["start_date", "end_date"]
        ids = ["permno", "barrid", "cusip", "ticker", "issue_name"]
        df = df.select(dates + ids + [col for col in df.columns if col not in (dates + ids)])

        return df
