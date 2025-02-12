import polars as pl

from silverfund.signals import Signal


class Score(pl.DataFrame):
    pass


def z_score(signal: Signal, col: str, over: str) -> Score:
    return Score(
        signal.with_columns(
            ((pl.col(col) - pl.col(col).mean().over(over)) / pl.col(col).std().over(over)).alias(
                "score"
            )
        )
    )
