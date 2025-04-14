from datetime import date

import polars as pl

from silverfund.data_access_layer_v2.schema.views import benchmark


def load(start_date: date, end_date: date) -> pl.DataFrame:
    return benchmark.filter(pl.col("date").is_between(start_date, end_date)).sort(["barrid", "date"]).collect()
