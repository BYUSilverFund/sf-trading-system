import polars as pl

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.scores import Score


class Alpha(pl.DataFrame):
    pass


def grindold_kahn(score: Score, ic: float = 0.05) -> Alpha:
    vols = dal.load_risk_forecasts(Interval.MONTHLY)
    return Alpha(
        score.join(other=vols, on=["date", "barrid"], how="inner").with_columns(
            ((ic * pl.col("total_risk") * pl.col("score")).alias("alpha"))
        )
    )
