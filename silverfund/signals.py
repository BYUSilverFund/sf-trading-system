import polars as pl


class Signal(pl.DataFrame):
    pass


def momentum(data: pl.DataFrame) -> Signal:
    return Signal(
        data.with_columns(pl.col("ret").log1p().alias("logret"))
        .with_columns(pl.col("logret").rolling_sum(11, min_periods=11).over("barrid").alias("mom"))
        .with_columns(pl.col("mom").shift(1).over("barrid"))
        .drop_nulls(subset="mom")
        .sort(["date", "barrid"])
    )
