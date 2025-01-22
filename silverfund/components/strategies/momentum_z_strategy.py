import polars as pl

from silverfund.components.enums import Interval
from silverfund.components.new_risk_model import NewRiskModel
from silverfund.components.optimizers import qp
from silverfund.components.strategies.strategy import Strategy


class MomentumZStrategy(Strategy):
    def __init__(self, interval: Interval):
        self._interval = interval
        self._window = {Interval.DAILY: 252, Interval.MONTHLY: 12}[self._interval]
        self._rolling_window = {Interval.DAILY: 230, Interval.MONTHLY: 11}[self._interval]
        self._skip = {Interval.DAILY: 22, Interval.MONTHLY: 1}[self._interval]

    def compute_signal(self, chunk: pl.DataFrame) -> pl.DataFrame:
        # Momentum signal formation
        chunk = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
            pl.col("logret").rolling_sum(window_size=self._rolling_window, min_periods=self._rolling_window, center=False).over("barrid").alias("mom")
        )

        # Momentum lag/skip
        chunk = chunk.with_columns(pl.col("mom").shift(self._skip).over("barrid"))

        # Non-null momentum
        chunk = chunk.drop_nulls(subset="mom")

        return chunk

    def compute_score(self, chunk: pl.DataFrame) -> pl.DataFrame:
        chunk = self.compute_signal(chunk)

        chunk = chunk.with_columns(((pl.col("mom") - pl.col("mom").mean()) / pl.col("mom").std()).alias("score"))

        return chunk

    def compute_alpha(self, chunk: pl.DataFrame) -> pl.DataFrame:
        chunk = self.compute_score(chunk)

        ic = 0.05

        chunk = chunk.with_columns((pl.col("total_risk") * pl.col("mom") * ic).alias("alpha"))

        return chunk

    def compute_portfolio(self, chunk: pl.DataFrame) -> list[pl.DataFrame]:
        chunk = self.compute_alpha(chunk)
        date_ = chunk["date"].max()
        barrids = chunk["barrid"].unique().to_list()

        # Load
        covariance_matrix = NewRiskModel(date_, barrids).load()
        alphas = chunk.select(["barrid", "alpha"])

        # Convert to numpy matrix and vecotr
        covariance_matrix = covariance_matrix.drop("barrid").to_numpy()
        alphas = alphas["alpha"].to_numpy()

        # Optimize
        weights = qp(alphas, covariance_matrix)

        # Package portfolio
        portfolio = chunk.with_columns(pl.Series(weights).alias("weight")).select(["date", "barrid", "weight"])
        return portfolio

    @property
    def window(self) -> int:
        return self._window
