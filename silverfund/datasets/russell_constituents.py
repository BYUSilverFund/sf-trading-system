import os
from pathlib import Path

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

    def get_benchmark_wts(self, id_scheme, dates=None, ids=None) -> pl.DataFrame:
        # Loading in Russell 3000 data from parquet
        russell_history = pl.read_parquet(self._file_path)
        # Restricting columns of dataframe to date, id scheme, and weights
        benchmark_wts = russell_history[["date", f"{id_scheme}", "r3000_wt"]]
        # Filtering to specified date range, if given; if not, all dates are given by default
        if dates:
            date_range = [SecurityMapping.to_datetime(date) for date in dates]
            start_date, end_date = date_range
            benchmark_wts = benchmark_wts.filter(
                (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
            )
        # Filtering to specified ids, if given; if not, all are given by default
        if ids:
            benchmark_wts = benchmark_wts.filter(pl.col(f"{id_scheme}").is_in(ids))
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
    print(rus.get_benchmark_wts("permno", ["2001-01-01", "2003-01-01"], [82766, 84161]))
