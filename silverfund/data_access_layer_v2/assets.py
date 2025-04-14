from datetime import date

import polars as pl

from silverfund.data_access_layer_v2.schema.views import assets_table, in_universe_assets


def load(start_date: date, end_date: date, in_universe: bool, columns: list[str]) -> pl.DataFrame:
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


def get_columns() -> str:
    return assets_table.columns()
