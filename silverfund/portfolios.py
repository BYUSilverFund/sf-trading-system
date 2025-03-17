import os
from datetime import date
from functools import partial
from typing import Protocol

import polars as pl
import ray
from ray.experimental import tqdm_ray
from tqdm import tqdm

import silverfund.data_access_layer as dal
from silverfund.alphas import Alpha
from silverfund.constraints import ConstraintConstructor
from silverfund.covariance_matrix import covariance_matrix_constructor
from silverfund.enums import Interval
from silverfund.optimizers import quadratic_program
from silverfund.records import Portfolio


class PortfolioConstructor(Protocol):
    """Protocol for functions that construct a Portfolio from alphas and constraints."""

    def __call__(
        self,
        period: date,
        barrids: list[str],
        alphas: Alpha,
        constraints: list[ConstraintConstructor],
    ) -> Portfolio: ...


def mean_variance_efficient(
    period: date,
    barrids: list[str],
    alphas: Alpha,
    constraints: list[ConstraintConstructor],
    gamma: float = 2.0,
) -> Portfolio:
    """Constructs a mean-variance efficient portfolio using quadratic optimization.

    This function builds an optimal portfolio by solving a quadratic programming
    problem that balances expected returns (alphas) and risk (covariance matrix)
    subject to given constraints.

    Args:
        period (date): The date for which the portfolio is constructed.
        barrids (list[str]): List of asset identifiers (barrids) included in the portfolio.
        alphas (Alpha): Expected returns for the assets.
        constraints (list[ConstraintConstructor]): List of constraints applied to the optimization.
        gamma (float, optional): Risk aversion parameter (default is 2.0).
                                 Higher values penalize risk more heavily.

    Returns:
        Portfolio: A Polars DataFrame wrapped in the Portfolio class,
                   containing 'date', 'barrid', and 'weight' columns.
    """

    # Get covariance matrix
    cov_mat = covariance_matrix_constructor(period, barrids)

    # Cast to numpy arrays
    alphas = alphas.to_vector()
    cov_mat = cov_mat.to_matrix()

    # Construct constraints
    constraints = [partial(constraint, date_=period, barrids=barrids) for constraint in constraints]

    # Find optimal weights
    weights = quadratic_program(alphas, cov_mat, constraints, gamma)

    portfolio = pl.DataFrame({"date": period, "barrid": barrids, "weight": weights})
    portfolio = portfolio.sort(["barrid", "date"])

    return Portfolio(portfolio)


def mve_sequential(
    start_date: date,
    end_date: date,
    alphas: Alpha,
    constraints: list[ConstraintConstructor],
    gamma: float = 2.0,
) -> pl.DataFrame:
    """
    Constructs mean-variance efficient (MVE) portfolios sequentially for each trading period.

    Args:
        start_date (date): The start date for portfolio construction.
        end_date (date): The end date for portfolio construction.
        alphas (Alpha): Expected returns or alpha signals for asset selection.
        constraints (list[ConstraintConstructor]): A list of portfolio constraints.
        gamma (float, optional): The risk aversion parameter. Default is 2.0.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the constructed portfolio with columns:
                      - 'date': The trading date.
                      - 'barrid': The asset identifier.
                      - 'weight': The portfolio weight assigned to each asset.
    """
    universe = dal.load_universe(
        interval=Interval.DAILY,
        start_date=start_date,
        end_date=end_date,
    )

    periods = universe["date"].unique().sort().to_list()

    portfolios = []
    for period in tqdm(periods, desc="Computing optimal portfolio weights"):
        # Get portfolio constructor parameters
        period_barrids = universe.filter(pl.col("date") == period)["barrid"].sort().to_list()
        period_alphas = Alpha(alphas.filter(pl.col("date") == period).sort(["barrid"]))

        # Construct period portfolio
        portfolio = mean_variance_efficient(
            period=period,
            barrids=period_barrids,
            alphas=period_alphas,
            constraints=constraints,
            gamma=gamma,
        )

        portfolios.append(portfolio)

    portfolios: pl.DataFrame = pl.concat(portfolios).sort(["barrid", "weight"])

    return portfolios


def mve_parallel(
    start_date: date,
    end_date: date,
    alphas: Alpha,
    constraints: list[ConstraintConstructor],
    gamma: float = 2.0,
    n_cpus: int | None = None,
) -> pl.DataFrame:
    """
    Constructs mean-variance efficient (MVE) portfolios in parallel using multiple CPUs.

    Args:
        start_date (date): The start date for portfolio construction.
        end_date (date): The end date for portfolio construction.
        alphas (Alpha): Expected returns or alpha signals for asset selection.
        constraints (list[ConstraintConstructor]): A list of portfolio constraints.
        gamma (float, optional): The risk aversion parameter. Default is 2.0.
        n_cpus (int, optional): Number of CPU cores to use for parallel processing. Defaults to all available cores.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the constructed portfolio with columns:
                      - 'date': The trading date.
                      - 'barrid': The asset identifier.
                      - 'weight': The portfolio weight assigned to each asset.
    """
    universe = dal.load_universe(
        interval=Interval.DAILY,
        start_date=start_date,
        end_date=end_date,
    )

    periods = universe["date"].unique().sort().to_list()

    # Set up ray
    n_cpus = n_cpus or os.cpu_count()
    n_cpus = min(len(periods), n_cpus)
    _ = ray.init(ignore_reinit_error=True, num_cpus=n_cpus)

    # Set up ray progress bar
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    progress_bar = remote_tqdm.remote(
        total=len(periods), desc=f"Computing portfolios with {n_cpus} cpus"
    )

    # Dispatch parallel tasks
    portfolio_futures = [
        construct_portfolio.remote(
            period=period,
            universe=universe,
            alphas=alphas,
            constraints=constraints,
            gamma=gamma,
            progress_bar=progress_bar,
        )
        for period in periods
    ]

    # Retrieve results
    portfolios = ray.get(portfolio_futures)

    # Shutdown ray and progress bar
    progress_bar.close.remote()
    ray.shutdown()

    # Concatenate
    portfolios: pl.DataFrame = pl.concat(portfolios).sort(["barrid", "weight"])

    return portfolios


@ray.remote
def construct_portfolio(
    period: date,
    universe: pl.DataFrame,
    alphas: Alpha,
    constraints: list[ConstraintConstructor],
    gamma: float = 2.0,
    progress_bar: tqdm_ray.tqdm | None = None,
):
    """
    Constructs a mean-variance efficient (MVE) portfolio for a given period.

    Args:
        period (date): The date for which the portfolio is being constructed.
        universe (pl.DataFrame): The universe of available assets for portfolio construction.
        alphas (Alpha): Expected returns or alpha signals for asset selection.
        constraints (list[ConstraintConstructor]): A list of portfolio constraints.
        gamma (float, optional): The risk aversion parameter. Default is 2.0.
        progress_bar (tqdm_ray.tqdm, optional): A Ray-based progress bar for tracking execution progress.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the constructed portfolio for the given period,
                      with 'barrid' and 'weight' columns.
    """
    # Get portfolio constructor parameters
    period_barrids = universe.filter(pl.col("date") == period)["barrid"].sort().to_list()
    period_alphas = Alpha(alphas.filter(pl.col("date") == period).sort(["barrid"]))

    # Construct period portfolio
    portfolio = mean_variance_efficient(
        period=period,
        barrids=period_barrids,
        alphas=period_alphas,
        constraints=constraints,
        gamma=gamma,
    )

    # Update progress bar
    if progress_bar is not None:
        progress_bar.update.remote(1)

    return portfolio
