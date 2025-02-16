from datetime import date
from functools import partial

from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import *
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

if __name__ == "__main__":
    # Date range
    start_date = date(1995, 7, 1)
    end_date = date(2024, 12, 31)

    # Define strategy
    strategy = Strategy(
        signal_constructor=momentum,
        score_constructor=partial(z_score, signal_col="mom"),
        alpha_constructor=grindold_kahn,
        portfolio_constructor=mean_variance_efficient,
        constraints=[full_investment, no_buying_on_margin, long_only, unit_beta],
    )

    # Instantiate backtester
    bt = Backtester(start_date=start_date, end_date=end_date, strategy=strategy)

    # Run sequentially
    pnl = bt.run_sequential()
    print("-" * 20 + " Asset Returns " + "-" * 20)
    print(pnl)

    pnl.write_parquet("research/backtest_example.parquet")
