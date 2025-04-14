import os
from typing import Optional

import polars as pl
from dotenv import load_dotenv


class Table:
    def __init__(self, name: str) -> None:
        self._name = name

        # Load environment variables
        load_dotenv(override=True)
        home, user = os.getenv("ROOT").split("/")[1:3]

        self._base_path = f"/{home}/{user}/groups/grp_quant/database"

    def file_path(self, year: Optional[int] = None) -> str:
        if year is None:
            return f"{self._base_path}/{self._name}/{self._name}_*.parquet"
        else:
            return f"{self._base_path}/{self._name}/{self._name}_{year}.parquet"

    def scan(self, year: Optional[int] = None) -> pl.LazyFrame:
        return pl.scan_parquet(self.file_path(year))

    def read(self, year: Optional[int] = None) -> pl.DataFrame:
        return pl.read_parquet(self.file_path(year))

    def columns(self) -> pl.DataFrame:
        pl.Config.set_tbl_rows(-1)
        schema = self.scan().collect_schema()
        df_str = str(
            pl.DataFrame(
                {
                    "column": list(schema.keys()),
                    "dtype": [str(t) for t in schema.values()],
                }
            )
        )
        pl.Config.set_tbl_rows(10)
        return df_str
