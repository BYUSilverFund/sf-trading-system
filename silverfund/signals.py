from typing import Protocol

import polars as pl

from silverfund.records import Signal


class SignalConstructor(Protocol):
    def __call__(self, data: pl.DataFrame) -> Signal: ...


def momentum(data: pl.DataFrame) -> Signal:
    signals = (
        data.with_columns(pl.col("ret").log1p().alias("logret"))
        .with_columns(pl.col("logret").rolling_sum(11, min_periods=11).over("barrid").alias("mom"))
        .with_columns(pl.col("mom").shift(1).over("barrid"))
        .select(["date", "barrid", "mom"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "mom")

def barraBAB(data: pl.DataFrame) -> Signal:
    signals = (
        data.with_columns(pl.col("predbeta") * -1)
        .select(["date", "barrid", "predbeta"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "predbeta")

def lowBeta(data: pl.DataFrame) -> Signal:
    signals = (
        data.with_columns(
            pl.col("predbeta")
            .qcut(10, labels=[str(x) for x in range(10)])
            .over("date")
            .alias("bin")
        )
        .with_columns(
            pl.when(pl.col("bin") == "0") 
            .then(pl.col("predbeta") * -1) 
            .otherwise(pl.lit(None))
            .alias("predbeta")
        )
        .select(["date", "barrid", "predbeta"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "predbeta")
