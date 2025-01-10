from datetime import date
from functools import partial

import polars as pl
from optimizers import decile_portfolio
from signals import reversal_signal

from shared.components import ChunkedData
from shared.datasets import AlpacaStock


def reversal_strategy(interval: str = "daily") -> list[pl.DataFrame]:
    """
    This is the script for the classic short term reversal trading strategy.
    It should be able to be passed to both a backtester and a live/paper trader.
    """
    match interval:
        case "daily":
            window = 23
        case "monthly":
            window = 1

    # Pull raw data
    raw_data = AlpacaStock(
        start_date=date(2020, 1, 1), end_date=date(2024, 12, 31), interval=interval
    ).load()

    # Create chunks
    chunked_data = ChunkedData(raw_data, window, ["date", "ticker", "ret"])

    # Apply signal transformations
    chunked_data.apply_signal_transform(partial(reversal_signal, interval=interval))
    chunked_data.remove_chunks()

    # Generate portfolios
    portfolios_list: list[list[pl.DataFrame]] = chunked_data.apply_portfolio_gen(
        partial(decile_portfolio, signal="rev")
    )

    # Long poor reversal, short good reversal
    portfolios_list = [
        pl.concat([portfolios[0], portfolios[9].with_columns(pl.col("weight") * -1)]).drop(
            ["bin", "rev"]
        )
        for portfolios in portfolios_list
    ]

    return portfolios_list
