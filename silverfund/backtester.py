import os
from datetime import date

import polars as pl
import ray
from ray.experimental import tqdm_ray
from tqdm import tqdm

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.records import Alpha, AssetReturns, Portfolio
from silverfund.strategies import Strategy


class Backtester:
    def __init__(self, interval: Interval, start_date: date, end_date: date, data: pl.DataFrame):
        self._interval = interval
        self._start_date = start_date
        self._end_date = end_date
        self._data = data

    def _compute_alphas(self, strategy: Strategy) -> Alpha:
        # Calculate signals, scores, and alphas
        signals = strategy.signal_constructor(self._data)
        scores = strategy.score_constructor(signals)
        alphas = strategy.alpha_constructor(scores)

        return Alpha(alphas)

    def _compute_forward_returns(self, portfolios: list[Portfolio]) -> AssetReturns:
        # Getting forward returns
        testing_data = (
            self._data.with_columns(pl.col("ret").shift(-1).over("barrid").alias("fwd_ret"))
            .select(["date", "barrid", "fwd_ret"])
            .sort(["barrid", "date"])
        )

        # Concatenate portfolios
        portfolios = pl.concat(portfolios)

        # Join forward returns on portfolios
        asset_returns = portfolios.join(testing_data, on=["barrid", "date"], how="left")
        asset_returns = asset_returns.sort(["barrid", "date"])

        return AssetReturns(asset_returns)

    def run_sequential(self, strategy: Strategy) -> AssetReturns:
        # Get universe
        universe = dal.load_universe(
            interval=self._interval,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        # Compute alphas
        alphas = self._compute_alphas(strategy)

        # Get periods
        periods = universe["date"].unique().sort().to_list()

        # Construct portfolios
        portfolios = [
            self.construct_portfolio(period, universe, alphas, strategy)
            for period in tqdm(periods, desc="Computing portfolios")
        ]

        return self._compute_forward_returns(portfolios)

    @staticmethod
    @ray.remote
    def construct_portfolio(
        period, universe, alphas, strategy, progress_bar: tqdm_ray.tqdm | None = None
    ):
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

        # Update progress bar
        if progress_bar is not None:
            progress_bar.update.remote(1)

        return portfolio

    def run_parallel(self, strategy: Strategy, n_cpus: int | None = None) -> AssetReturns:
        # Get universe
        universe = dal.load_universe(
            interval=self._interval,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        # Compute alphas
        alphas = self._compute_alphas(strategy)

        # Get periods
        periods = universe["date"].unique().sort().to_list()

        # Set up ray
        n_cpus = n_cpus or os.cpu_count()
        n_cpus = min(len(periods), n_cpus)
        ray.init(ignore_reinit_error=True, num_cpus=n_cpus)

        # Set up ray progress bar
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        progress_bar = remote_tqdm.remote(
            total=len(periods), desc=f"Computing portfolios with {n_cpus} cpus"
        )

        # Dispatch parallel tasks
        portfolio_futures = [
            self.construct_portfolio.remote(period, universe, alphas, strategy, progress_bar)
            for period in periods
        ]

        # Retrieve results
        portfolios = ray.get(portfolio_futures)

        # Shutdown ray and progress bar
        progress_bar.close.remote()
        ray.shutdown()

        return self._compute_forward_returns(portfolios)
