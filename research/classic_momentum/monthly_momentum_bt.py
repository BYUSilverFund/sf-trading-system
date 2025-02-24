from datetime import date
from functools import partial

from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import *
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

if __name__ == "__main__":
    # Date range
    start_date = date(2001, 1, 1)
    end_date = date(2020, 12, 31)
    interval = Interval.MONTHLY

    # Define strategy
    strategy = Strategy(
        signal_constructor=momentum,
        score_constructor=partial(z_score, signal_col="mom"),
        alpha_constructor=partial(grindold_kahn, interval=interval),
        portfolio_constructor=mean_variance_efficient,
        constraints=[
            full_investment,
            no_buying_on_margin,
            long_only,
            partial(unit_beta, interval=interval),
        ],
    )

    universe = dal.load_universe(interval=interval, start_date=start_date, end_date=end_date)

    training_data = universe.join(
        dal.load_barra_returns(interval=interval, start_date=start_date, end_date=end_date),
        on=["date", "barrid"],
        how="left",
    ).sort(["barrid", "date"])

    # Instantiate backtester
    bt = Backtester(interval=interval, start_date=start_date, end_date=end_date, data=training_data)

    # Run sequentially
    asset_returns = bt.run_sequential(strategy)
    print("-" * 20 + " Asset Returns " + "-" * 20)
    print(asset_returns)

    asset_returns.write_parquet("research/classic_momentum/results/monthly_momentum_bt.parquet")
