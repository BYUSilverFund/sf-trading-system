import polars as pl

from silverfund.enums import Interval
from silverfund.new_risk_model import NewRiskModel
from silverfund.optimizers import qp
from silverfund.optimizers.new_constraints import Constraint
from silverfund.strategies.strategy import Strategy


class ReversalZStrategy(Strategy):
    def __init__(self, interval: Interval):
        self._interval = interval
        self._window = {Interval.DAILY: 23, Interval.MONTHLY: 2}[self._interval]
        self._rolling_window = {Interval.DAILY: 22, Interval.MONTHLY: 1}[self._interval]
        self._skip = {Interval.DAILY: 1, Interval.MONTHLY: 1}[self._interval]

    def compute_signal(self, chunk: pl.DataFrame) -> pl.DataFrame:
        # Momentum signal formation
        # chunk = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
        #     pl.col("logret").rolling_sum(window_size=self._rolling_window, min_periods=self._rolling_window, center=False).over("barrid").alias("mom")
        # )

        chunk = chunk.with_columns(
            pl.col("log_spec_ret")
            .rolling_sum(
                window_size=self._rolling_window, min_periods=self._rolling_window, center=False
            )
            .over("barrid")
            .alias("rev")
        )

        # reversal lag/skip
        chunk = chunk.with_columns(pl.col("rev").shift(self._skip).over("barrid"))

        # Non-null reversal values
        chunk = chunk.drop_nulls(subset="rev")

        return chunk

    def compute_score(self, chunk: pl.DataFrame) -> pl.DataFrame:
        chunk = self.compute_signal(chunk)

        # chunk = chunk.with_columns(((pl.col("rev") - pl.col("rev").mean()) / pl.col("rev").std()).alias("score"))
        chunk = chunk.with_columns(
            ((pl.col("rev") - pl.col("rev").mean()) / pl.col("rev").std()).alias("score")
        )

        return chunk

    def compute_alpha(self, chunk: pl.DataFrame) -> pl.DataFrame:
        chunk = self.compute_score(chunk)

        ic = 0.05

        # chunk = chunk.with_columns((pl.col("total_risk") * - pl.col("rev") * ic).alias("alpha"))
        # chunk = chunk.with_columns((pl.col("total_risk") * pl.col("rev") * ic).alias("alpha"))
        # chunk = chunk.with_columns((pl.col("total_risk") * - pl.col("score") * ic).alias("alpha"))
        # chunk = chunk.with_columns((pl.col("total_risk") * 1 * ic).alias("alpha"))
        chunk = chunk.with_columns((pl.col("total_risk") * -1 * ic).alias("alpha"))

        return chunk

    def compute_portfolio(
        self, chunk: pl.DataFrame, constraints: list[Constraint]
    ) -> list[pl.DataFrame]:
        chunk = self.compute_alpha(chunk)
        date_lag = chunk["date"].max()
        barrids = chunk["barrid"].unique().to_list()

        # Load covariance matrix on prior date
        covariance_matrix = NewRiskModel(date_lag, barrids).load().drop("barrid")
        covariance_matrix = covariance_matrix.to_numpy()
        # print(chunk)
        # Optimize
        portfolio = qp(chunk, covariance_matrix, constraints)

        return portfolio

    @property
    def window(self) -> int:
        return self._window
