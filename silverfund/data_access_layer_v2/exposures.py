from datetime import date
from typing import Optional

import polars as pl

from silverfund.data_access_layer_v2.schema.tables import exposures_table


def load(start_date: date, end_date: date, barrids: Optional[list[str]] = None) -> pl.DataFrame:
    df = exposures_table.scan().filter(pl.col("date").is_between(start_date, end_date))

    if barrids is not None:
        df = df.filter(pl.col("barrid").is_in(barrids))

    return df.sort(["barrid", "date"]).collect()


def get_columns() -> str:
    return exposures_table.columns()
