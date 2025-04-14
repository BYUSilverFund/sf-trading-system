from datetime import date

import polars as pl

from silverfund.data_access_layer_v2.schema.tables import covariances_table


def load(start_date: date, end_date: date) -> pl.DataFrame:
    return covariances_table.scan().filter(pl.col("date").is_between(start_date, end_date)).sort("date").collect()


def get_columns() -> str:
    return covariances_table.columns()
