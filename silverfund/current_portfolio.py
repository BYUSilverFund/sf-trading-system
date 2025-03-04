from datetime import date, timedelta
from functools import partial

import exchange_calendars as xcals
import polars as pl

import silverfund.data_access_layer as dal
from silverfund.alphas import grindold_kahn
from silverfund.constraints import full_investment, long_only, no_buying_on_margin, unit_beta
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.records import Alpha
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy


def get_last_market_date(current_date: date) -> date:
    """
    Get the last trading day before the given date.

    This function retrieves the previous market date based on the
    New York Stock Exchange (XNYS) trading calendar.

    Args:
        current_date (date): The reference date to find the previous trading day.

    Returns:
        date: The last trading day before the given date.
    """
    # Load market calendar
    market_calendar = (
        pl.DataFrame(xcals.get_calendar("XNYS").schedule)
        .with_columns(pl.col("close").dt.date())
        .select(pl.col("close").alias("date"))
        .with_columns(pl.col("date").shift(1).alias("prev_date"))
    )

    # Get previous date
    prev_date = market_calendar.filter(pl.col("date").eq(current_date))["prev_date"].max()

    return prev_date


def get_current_portfolio(
    strategy: Strategy,
    interval: Interval,
    look_back: int,
    current_date: date | None = None,
) -> pl.DataFrame:
    """
    Construct the current portfolio based on a given strategy.

    This function loads market data, computes trading signals, converts them
    into scores, generates alphas, and constructs a portfolio using the
    specified strategy and constraints.

    Args:
        strategy (Strategy): The trading strategy defining signal, score, alpha, and portfolio construction methods.
        interval (Interval): The data frequency (e.g., daily, weekly).
        look_back (int): The number of days to look back when retrieving market data.
        current_date (date | None, optional): The reference date for portfolio construction. Defaults to today's date.

    Returns:
        pl.DataFrame: A DataFrame representing the constructed portfolio.
    """
    # Data parameters
    end_date = date.today()
    start_date = end_date - timedelta(days=look_back)

    # Get universe
    universe = dal.load_universe(interval=interval, start_date=start_date, end_date=end_date)

    # Get returns data
    data = universe.join(
        dal.load_barra_returns(interval=interval, start_date=start_date, end_date=end_date).sort(
            ["barrid", "date"]
        ),
        on=["barrid", "date"],
        how="left",
    )

    # Construct signals, scores, and alphas
    signals = strategy.signal_constructor(data)
    scores = strategy.score_constructor(signals)
    alphas = strategy.alpha_constructor(scores, interval)

    # Get previous market date
    current_date = current_date or date.today()
    prev_date = get_last_market_date(current_date)

    # Filter to current alphas
    current_alphas_df = alphas.filter(pl.col("date") == prev_date).sort(["barrid", "date"])

    # Get current alphas and barrids
    current_alphas = Alpha(current_alphas_df)
    barrids = current_alphas_df["barrid"].to_list()

    # Construct current portfolio
    portfolio = strategy.portfolio_constructor(
        period=prev_date,
        barrids=barrids,
        alphas=current_alphas,
        constraints=strategy.constraints,
    )

    return portfolio


if __name__ == "__main__":
    # Specify rebalancing interval
    interval = Interval.DAILY

    # Specify strategy
    strategy = Strategy(
        signal_constructor=momentum,
        score_constructor=partial(z_score, signal_col="mom"),
        alpha_constructor=grindold_kahn,
        portfolio_constructor=mean_variance_efficient,
        constraints=[
            full_investment,
            no_buying_on_margin,
            long_only,
            partial(unit_beta, interval=interval),
        ],
    )

    # Get current portfolio
    current_portfolio = get_current_portfolio(
        strategy=strategy,
        interval=interval,
        look_back=365,
        current_date=date(2024, 11, 29),
    ).drop("date")

    # Log
    print("-" * 20 + " Current Portfolio " + "-" * 20)
    print(current_portfolio)

    # Write to csv
    current_portfolio.write_csv("current_portfolio.csv")
