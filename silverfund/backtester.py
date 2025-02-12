from datetime import date

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.strategies import Strategy


class Backtester:
    def __init__(self, start_date: date, end_date: date, strategy: Strategy):
        self._start_date = start_date
        self._end_date = end_date
        self._strategy = strategy

    def run(self, data: pl.DataFrame) -> pl.DataFrame:

        historical_data = (
            dal.load_barra_returns(start_date=self._start_date, end_date=self._end_date)
            .with_columns(pl.col("ret").shift(-1).alias("fwd_ret"))
            .select(["date", "barrid", "fwd_ret"])
            .drop_nulls()
        )

        universe = dal.load_monthly_universe(start_date=self._start_date, end_date=self._end_date)

        returns = universe.join(historical_data, on=["date", "barrid"], how="inner")
        returns = returns.sort(["date", "barrid"])

        signals = self._strategy.signal_fn(data)
        scores = self._strategy.score_fn(signals)
        alphas = self._strategy.alpha_fn(scores)

        cov_mat = pl.DataFrame()
        constraints = self._strategy.constraints
        portfolios = self._strategy.optimizer(alphas, cov_mat, constraints)

        pnl = returns.join(portfolios, on=["barrid", "date"], how="left")

        return pnl
