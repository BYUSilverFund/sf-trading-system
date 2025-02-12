from typing import Protocol

import polars as pl

from silverfund.signals import Signal


class Score(pl.DataFrame):
    def __init__(self, scores: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "score"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "score": pl.Float64,
        }

        # Check if all required columns exist
        if set(expected_order) != set(scores.columns):
            missing = set(expected_order) - set(scores.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure correct column types
        for col, dtype in valid_schema.items():
            if scores.schema[col] != dtype:
                raise ValueError(
                    f"Column {col} has incorrect type: {scores.schema[col]}, expected: {dtype}"
                )

        # Reorder columns
        scores = scores.select(expected_order)

        # Initialize
        super().__init__(scores)


class ScoreConstructor(Protocol):
    def __call__(self, signals: Signal, col: str, over: str) -> Score: ...


def z_score(signals: Signal, signal_col: str) -> Score:
    return Score(
        signals.with_columns(
            (
                (pl.col(signal_col) - pl.col(signal_col).mean().over("date"))
                / pl.col(signal_col).std().over("date")
            ).alias("score")
        )
        .select(["date", "barrid", "score"])
        .sort(["barrid", "date"])
    )
