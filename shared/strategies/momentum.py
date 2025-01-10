from datetime import date
from functools import partial

import polars as pl
from optimizers import decile_portfolio
from signals import momentum_signal

from shared.components import ChunkedData
from shared.datasets import AlpacaStock


def momentum_strategy(interval: str = "daily") -> list[pl.DataFrame]:
    """
    This is the script for the classic momentum trading strategy.
    It should be able to be passed to both a backtester and a live/paper trader.
    """

    match interval:
        case "daily":
            window = 231
        case "monthly":
            window = 12

    # Pull raw data
    raw_data = AlpacaStock(
        start_date=date(2020, 1, 1), end_date=date(2024, 12, 31), interval=interval
    ).load()

    # Create chunks
    chunked_data = ChunkedData(data=raw_data, window=window, columns=["date", "ticker", "ret"])

    # Apply signal transformations
    chunked_data.apply_signal_transform(partial(momentum_signal, interval=interval))
    chunked_data.remove_chunks()

    # Generate portfolios
    portfolios_list: list[list[pl.DataFrame]] = chunked_data.apply_portfolio_gen(
        partial(decile_portfolio, signal="mom")
    )

    # Long good momentum, short poor momentum
    portfolios_list = [
        pl.concat([portfolios[0].with_columns(pl.col("weight") * -1), portfolios[9]]).drop(
            ["bin", "mom"]
        )
        for portfolios in portfolios_list
    ]

    return portfolios_list
