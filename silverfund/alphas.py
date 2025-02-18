from typing import Protocol

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.records import Alpha, Score


class AlphaConstructor(Protocol):
    def __call__(self, score: Score) -> Alpha: ...


def grindold_kahn(scores: Score, interval: Interval, ic: float = 0.05) -> Alpha:
    start_date = scores["date"].min()
    end_date = scores["date"].max()

    vols = dal.load_total_risk(interval, start_date, end_date).with_columns(pl.col("total_risk"))

    return Alpha(
        scores.join(other=vols, on=["date", "barrid"], how="left")
        .with_columns(((ic * pl.col("total_risk") * pl.col("score")).alias("alpha")))
        .fill_null(0)
        .select(["date", "barrid", "alpha"])
        .sort(["barrid", "date"])
    )


def static_alpha(scores: Score, value: float) -> Alpha:
    return Alpha(scores.with_columns(pl.lit(value).cast(pl.Float64).alias("alpha")).drop("score"))
