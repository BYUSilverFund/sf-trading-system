import os
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class SecurityMapping:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        user = os.getenv("ROOT").split("/")[2]
        root_dir = Path(f"/home/{user}")

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

    def to_datetime(date) -> datetime:
        for fmt in ("%Y-%m-%d", "%Y"):  # Specify possible formats
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date format not recognized: {date}. Try '%Y-%m-%d' or '%Y'.")
