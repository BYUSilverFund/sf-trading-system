from typing import Protocol

import polars as pl


class Signal(pl.DataFrame):
    def __init__(self, signals: pl.DataFrame, signal_name: str) -> None:
        expected_order = ["date", "barrid", signal_name]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            signal_name: pl.Float64,
        }

        # Check if all required columns exist
        if set(expected_order) != set(signals.columns):
            missing = set(expected_order) - set(signals.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure correct column types
        for col, dtype in valid_schema.items():
            if signals.schema[col] != dtype:
                raise ValueError(
                    f"Column {col} has incorrect type: {signals.schema[col]}, expected: {dtype}"
                )

        # Reorder columns
        signals = signals.select(expected_order)

        # Initialize
        super().__init__(signals)


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
