from typing import Protocol

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.scores import Score


class Alpha(pl.DataFrame):
    def __init__(self, alphas: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "alpha"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "alpha": pl.Float64,
        }

        # Check if all required columns exist
        if set(expected_order) != set(alphas.columns):
            missing = set(expected_order) - set(alphas.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure correct column types
        for col, dtype in valid_schema.items():
            if alphas.schema[col] != dtype:
                raise ValueError(
                    f"Column {col} has incorrect type: {alphas.schema[col]}, expected: {dtype}"
                )

        # Reorder columns
        alphas = alphas.select(expected_order)

        # Initialize
        super().__init__(alphas)

    def to_vector(self):
        return self.select("alpha").to_numpy()


class AlphaConstructor(Protocol):
    def __call__(self, score: Score) -> Alpha: ...


def grindold_kahn(scores: Score, interval: Interval, ic: float = 0.05) -> Alpha:
    start_date = scores["date"].min()
    end_date = scores["date"].max()

    vols = dal.load_total_risk(interval, start_date, end_date).with_columns(
        pl.col("total_risk") * 100
    )  # put in percent space

    return Alpha(
        scores.join(other=vols, on=["date", "barrid"], how="left")
        .with_columns(((ic * pl.col("total_risk") * pl.col("score")).alias("alpha")))
        .fill_null(0)
        .select(["date", "barrid", "alpha"])
        .sort(["barrid", "date"])
    )


def static_alpha(scores: Score, value: float) -> Alpha:
    return Alpha(scores.with_columns(pl.lit(value).cast(pl.Float64).alias("alpha")).drop("score"))
