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
