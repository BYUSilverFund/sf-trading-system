import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv
from security_mapping import SecurityMapping

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

    def get_benchmark_wts(
        self,
        id_scheme: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        id_list: Optional[list] = None,
    ) -> pl.DataFrame:
        # Loading in Russell 3000 data from parquet
        russell_history = pl.read_parquet(self._file_path)
        # Restricting columns of dataframe to date, id scheme, and weights
        benchmark_wts = russell_history[["date", f"{id_scheme}", "r3000_wt"]]
        # Filtering to specified date range, if given; if not, all dates are given by default
        if start_date:
            start_date = SecurityMapping.to_datetime(start_date)
            benchmark_wts = benchmark_wts.filter(pl.col("date") >= start_date)
        if end_date:
            end_date = SecurityMapping.to_datetime(end_date)
            benchmark_wts = benchmark_wts.filter(pl.col("date") <= end_date)
        # Filtering to specified ids, if given; if not, all are given by default
        if id_list:
            benchmark_wts = benchmark_wts.filter(pl.col(f"{id_scheme}").is_in(id_list))
        return benchmark_wts

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        df = df.with_columns(
            pl.col("date").dt.date(),
            pl.col("obsdate").dt.date(),
            pl.col("enddate").dt.date(),
        )

        return df


if __name__ == "__main__":
    rus = RussellConstituents()
    print(rus.get_benchmark_wts("permno", "2021-01-01", "2024-12-31", [15857, 16454, 92679]))
