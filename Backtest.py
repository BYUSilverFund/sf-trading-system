import os
from datetime import date
from functools import partial
from pathlib import Path

import silverfund.data_access_layer as dal
from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import full_investment, long_only, no_buying_on_margin, unit_beta
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

if __name__ == "__main__":
    # Define backtest period
    start_date = date(1995, 7, 31)
    end_date = date(2024, 12, 31)
    interval = Interval.DAILY

    # Load universe and returns data
    universe = dal.load_universe(interval=interval, start_date=start_date, end_date=end_date)
    returns_data = dal.load_barra_returns(
        interval=interval, start_date=start_date, end_date=end_date
    )

    # Prepare dataset
    training_data = universe.join(returns_data, on=["date", "barrid"], how="left").sort(
        ["barrid", "date"]
    )

    # Check what momentum() returns
    test_signals = momentum(training_data)
    print("Momentum Signal Columns:", test_signals.columns)  # Debugging step

    # Adjust signal column name if needed
    if "momentum_signal" not in test_signals.columns:
        momentum_col = test_signals.columns[-1]  # Assuming last column is the signal
    else:
        momentum_col = "momentum_signal"

    # Define momentum strategy
    strategy = Strategy(
        signal_constructor=momentum,
        score_constructor=partial(z_score, signal_col=momentum_col),  # Dynamically adjusting
        alpha_constructor=partial(grindold_kahn, interval=interval),  # Alpha transformation
        portfolio_constructor=mean_variance_efficient,  # Portfolio optimization
        constraints=[
            full_investment,
            no_buying_on_margin,
            long_only,
            partial(unit_beta, interval=interval),  # Market-neutral portfolio constraint
        ],
    )

    # Instantiate backtester
    bt = Backtester(interval=interval, start_date=start_date, end_date=end_date, data=training_data)

    # Run strategy in parallel
    asset_returns = bt.run_parallel(strategy)
    print("-" * 20 + " Asset Returns " + "-" * 20)
    print(asset_returns)

    # Save backtest results
    results_folder = Path("research/momentum_strategy/results")
    os.makedirs(results_folder, exist_ok=True)
    asset_returns.write_parquet(results_folder / "momentum_backtest.parquet")
