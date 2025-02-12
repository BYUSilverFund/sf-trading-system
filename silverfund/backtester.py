from datetime import date

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.strategies import Strategy


class Backtester:
    def __init__(self, start_date: date, end_date: date, strategy: Strategy):
        self._start_date = start_date
        self._end_date = end_date
        self._strategy = strategy

    def run(self) -> pl.DataFrame:
        universe = dal.load_monthly_universe(start_date=self._start_date, end_date=self._end_date)

        training_data = (
            dal.load_barra_returns(start_date=self._start_date, end_date=self._end_date)
            .join(universe, on=["date", "barrid"], how="inner")
            .sort(["barrid", "date"])
        )

        testing_data = (
            training_data.with_columns(pl.col("ret").shift(-1).alias("fwd_ret"))
            .select(["date", "barrid", "fwd_ret"])
            .drop_nulls()
            .sort(["barrid", "date"])
        )
        print("DATA", training_data)
        signals = self._strategy.signal_fn(training_data)
        print("SIGNALS", signals)
        scores = self._strategy.score_fn(signals)
        print("SCORES", scores)
        alphas = self._strategy.alpha_fn(scores)
        print("ALPHAS", alphas)

        cov_mat = pl.DataFrame()
        constraints = self._strategy.constraints
        portfolios = self._strategy.optimizer(alphas, cov_mat, constraints)

        pnl = testing_data.join(portfolios, on=["barrid", "date"], how="left")

        return pnl
