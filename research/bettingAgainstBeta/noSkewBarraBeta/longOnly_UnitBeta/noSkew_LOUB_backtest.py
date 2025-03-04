from datetime import date
from functools import partial

from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import *
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import barraBAB_noskew
from silverfund.strategies import Strategy

if __name__ == "__main__":
    # Date range
    start_date = date(1996, 1, 1)
    end_date = date(2023, 12, 31)
    interval = Interval.MONTHLY

    # Define strategy
    strategy = Strategy(
        signal_constructor=barraBAB_noskew,
        score_constructor=partial(z_score, signal_col="noskew_predbeta"),
        alpha_constructor=partial(grindold_kahn, interval=interval),
        portfolio_constructor=mean_variance_efficient,
        constraints=[
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
    
    barra_total_risk = dal.load_total_risk(
        interval=interval, start_date=start_date, end_date=end_date, quiet=False
    )
    
    training_data = training_data.join(
        barra_total_risk.select(["date", "barrid", "predbeta"]),  # Select only relevant columns
        on=["date", "barrid"],
        how="left"
    )

    # Instantiate backtester
    bt = Backtester(interval=interval, start_date=start_date, end_date=end_date, data=training_data)

    # Run backtest
    pnl = bt.run_parallel(strategy)
    print("-" * 20 + " Asset Returns " + "-" * 20)
    print(pnl)

    pnl.write_parquet("noSkew_LOUB_backtest.parquet")