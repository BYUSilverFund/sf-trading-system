from datetime import date

import polars as pl

from silverfund.data_access_layer_v2.schema.views import crsp_monthly_clean, crsp_monthly_table


def load(start_date: date, end_date: date, columns: list[str]) -> pl.DataFrame:
    return (
        crsp_monthly_clean.filter(pl.col("date").is_between(start_date, end_date))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_columns() -> str:
    return crsp_monthly_table.columns()
