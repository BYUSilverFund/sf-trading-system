from typing import Protocol

import polars as pl

from silverfund.records import Score, Signal


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
    
def uniform_score(signals: Signal, signal_col: str) -> Score:
    return Score(
        signals.with_columns(
            pl.when(pl.col(signal_col).is_not_null())  
            .then(pl.lit(1.0)) 
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64)
            .alias("score")
        )
        .select(["date", "barrid", "score"])
        .sort(["barrid", "date"])
    )