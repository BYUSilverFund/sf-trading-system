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


def idio_mom(data: pl.DataFrame) -> Signal:
    signals = (
        data.with_columns(pl.col("idio_mom_5f"))
        .select(["date", "barrid", "idio_mom_5f"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "idio_mom_5f")
