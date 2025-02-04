import polars as pl

from silverfund.enums import Interval
from silverfund.new_risk_model import NewRiskModel
from silverfund.optimizers import qp
from silverfund.optimizers.new_constraints import Constraint
from silverfund.strategies.strategy import Strategy


class MomentumZStrategy(Strategy):
    def __init__(self, interval: Interval):
        self._interval = interval
        self._window = {Interval.DAILY: 252, Interval.MONTHLY: 13}[self._interval]
        self._rolling_window = {Interval.DAILY: 230, Interval.MONTHLY: 11}[self._interval]
        self._lag = {Interval.DAILY: 22, Interval.MONTHLY: 2}[self._interval]

    def compute_signal(self, chunk: pl.DataFrame) -> pl.DataFrame:
        # Momentum signal formation
        chunk = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
            pl.col("logret")
            .rolling_sum(
                window_size=self._rolling_window, min_periods=self._rolling_window, center=False
            )
            .over("barrid")
            .alias("mom")
        )

        # Momentum lag/skip
        chunk = chunk.with_columns(pl.col("mom").shift(self._lag).over("barrid").alias("mom_lag"))

        # Lag total_risk
        chunk = chunk.with_columns(
            pl.col("total_risk").shift(1).over("barrid").alias("total_risk_lag")
        )

        # Lag predbeta
        chunk = chunk.with_columns(pl.col("predbeta").shift(1).over("barrid").alias("predbeta_lag"))

        # Non-null momentum
        chunk = chunk.drop_nulls(subset="mom_lag")

        return chunk

    def compute_score(self, chunk: pl.DataFrame) -> pl.DataFrame:
        chunk = self.compute_signal(chunk)

        # Z-score the momentum signal
        chunk = chunk.with_columns(
            ((pl.col("mom_lag") - pl.col("mom_lag").mean()) / pl.col("mom_lag").std()).alias(
                "score"
            )
        )

        return chunk

    def compute_alpha(self, chunk: pl.DataFrame) -> pl.DataFrame:
        chunk = self.compute_score(chunk)

        ic = 0.05

        chunk = chunk.with_columns(
            (pl.col("total_risk_lag") * pl.col("mom_lag") * ic).alias("alpha")
        )

        return chunk

    def compute_portfolio(
        self, chunk: pl.DataFrame, constraints: list[Constraint]
    ) -> list[pl.DataFrame]:
        chunk = self.compute_alpha(chunk)
        date_lag = chunk["date_lag_1d"].max()
        barrids = chunk["barrid"].unique().to_list()

        # Load covariance matrix on prior date
        covariance_matrix = NewRiskModel(date_lag, barrids).load().drop("barrid")
        covariance_matrix = covariance_matrix.to_numpy()

        # Optimize
        portfolio = qp(chunk, covariance_matrix, constraints)

        return portfolio

    @property
    def window(self) -> int:
        return self._window
