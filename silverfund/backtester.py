from datetime import date

import polars as pl
from tqdm import tqdm

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.records import Alpha, AssetReturns
from silverfund.strategies import Strategy


class Backtester:
    def __init__(self, interval: Interval, start_date: date, end_date: date, data: pl.DataFrame):
        self._interval = interval
        self._start_date = start_date
        self._end_date = end_date
        self._data = data

    def run_sequential(self, strategy: Strategy) -> AssetReturns:
        # Universe, training_data, and testing_data will all be parameters in the future.
        universe = dal.load_universe(
            interval=self._interval, start_date=self._start_date, end_date=self._end_date
        )

        testing_data = (
            self._data.with_columns(pl.col("ret").shift(-1).over("barrid").alias("fwd_ret"))
            .select(["date", "barrid", "fwd_ret"])
            .sort(["barrid", "date"])
        )

        # Calculate signals, scores, and alphas
        signals = strategy.signal_constructor(self._data)
        scores = strategy.score_constructor(signals)
        alphas = strategy.alpha_constructor(scores)

        # Get unique periods
        periods = universe["date"].unique().sort().to_list()
        portfolios = []
        for period in tqdm(periods, desc="Computing portfolios"):

            # Get portfolio constructor parameters
            period_barrids = universe.filter(pl.col("date") == period)["barrid"].sort().to_list()
            period_alphas = Alpha(alphas.filter(pl.col("date") == period).sort(["barrid"]))

            # Construct period portfolio
            portfolio = strategy.portfolio_constructor(
                period=period,
                barrids=period_barrids,
                alphas=period_alphas,
                constraints=strategy.constraints,
            )
            portfolios.append(portfolio)

        # Concatenate portfolios
        portfolios = pl.concat(portfolios)

        # Join forward returns on portfolios
        asset_returns = portfolios.join(testing_data, on=["barrid", "date"], how="left")
        asset_returns = asset_returns.sort(["barrid", "date"])

        return AssetReturns(asset_returns)
