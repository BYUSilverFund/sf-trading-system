import os
from datetime import date
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


assets_table = Table("assets")

russell_rebalance_dates = (
    assets_table.scan()
    # Standard filters
    .filter(pl.col("barrid").eq(pl.col("rootid")))
    .filter(pl.col("iso_country_code").eq("USA"))
    # Russell constituency filter
    .filter(pl.col("russell_1000") | pl.col("russell_2000"))
    # Create rebalance column
    .select("date", pl.lit(True).alias("russell_rebalance"))
    .unique()
)

in_universe_assets = (
    assets_table.scan()
    # Standard filters
    .filter(pl.col("barrid").eq(pl.col("rootid"))).filter(pl.col("iso_country_code").eq("USA"))
    # Join rebalance dates
    .join(russell_rebalance_dates, on="date", how="left")
    # Fill nulls with false on rebalance dates
    .with_columns(
        pl.when(pl.col("russell_rebalance")).then(
            pl.col("russell_1000", "russell_2000").fill_null(False)
        )
    )
    # Sort before forward fill
    .sort(["barrid", "date"])
    # Forward fill
    .with_columns(
        pl.col("ticker", "russell_1000", "russell_2000")
        .fill_null(strategy="forward")
        .over("barrid")
    )
    # Russell constituency filter
    .filter(pl.col("russell_1000") | pl.col("russell_2000"))
    # Drop russell_rebalance column
    .drop("russell_rebalance")
)


def load_assets(
    start_date: date, end_date: date, in_universe: bool, columns: list[str]
) -> pl.DataFrame:
    if in_universe:
        return (
            in_universe_assets.filter(pl.col("date").is_between(start_date, end_date))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )
    else:
        return (
            assets_table.scan()
            .filter(pl.col("date").is_between(start_date, end_date))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def get_assets_columns() -> str:
    return assets_table.columns()
