from datetime import date

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


def grindold_kahn(score: Score, ic: float = 0.05) -> Alpha:
    vols = dal.load_risk_forecasts(Interval.MONTHLY)
    return Alpha(
        score.join(other=vols, on=["date", "barrid"], how="left")
        .with_columns(((ic * pl.col("total_risk") * pl.col("score")).alias("alpha")))
        .fill_null(0)
        .select(["date", "barrid", "alpha"])
        .sort(["date", "barrid"])
    )
