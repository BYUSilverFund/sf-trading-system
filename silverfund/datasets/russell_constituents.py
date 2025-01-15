import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from silverfund.database import Database


class RussellConstituents:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        user = os.getenv("ROOT").split("/")[2]
        root_dir = Path(f"/home/{user}")

        self._file_path = root_dir / "groups" / "grp_quant" / "data" / "russell_history.parquet"

    def load_all(self) -> pl.DataFrame:

        return self.clean(pl.read_parquet(self._file_path))

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        df = df.with_columns(
            pl.col("date").dt.date(),
            pl.col("obsdate").dt.date(),
            pl.col("enddate").dt.date(),
        )

        return df


if __name__ == "__main__":
    rus = RussellConstituents().load()
    print(rus)
