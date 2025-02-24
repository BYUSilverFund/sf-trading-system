from multiprocessing import Pool
from datetime import date
from functools import partial
import pandas as pd

from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import *
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import barraBAB
from silverfund.strategies import Strategy

def run_backtest_for_year(year: int, interval: Interval, start_date: date, end_date: date):
    year_start_date = date(year, 1, 1)
    year_end_date = date(year, 12, 31)

    strategy = Strategy(
        signal_constructor=barraBAB,
        score_constructor=partial(z_score, signal_col="predbeta"),
        alpha_constructor=partial(grindold_kahn, interval=interval),
        portfolio_constructor=mean_variance_efficient,
        constraints=[
            partial(zero_beta, interval=interval),
        ],
    )

    universe = dal.load_universe(interval=interval, start_date=start_date, end_date=end_date)
    training_data = universe.join(
        dal.load_barra_returns(interval=interval, start_date=year_start_date, end_date=year_end_date),
        on=["date", "barrid"],
        how="left",
    ).sort(["barrid", "date"])
    
    barra_total_risk = dal.load_total_risk(
        interval=interval, start_date=year_start_date, end_date=year_end_date, quiet=False
    )
    
    training_data = training_data.join(
        barra_total_risk.select(["date", "barrid", "predbeta"]),  # Select only relevant columns
        on=["date", "barrid"],
        how="left"
    )

    bt = Backtester(interval=interval, start_date=year_start_date, end_date=year_end_date, data=training_data)

    pnl = bt.run_sequential(strategy)
    print(f"Backtest completed for {year}")
    
    pnl.write_parquet(f"results/barraBAB_backtest_{year}.parquet")

def run_parallel_backtests():
    # Date range
    start_date = date(1996, 1, 1)
    end_date = date(2023, 12, 31)
    interval = Interval.DAILY

    with Pool() as pool:
        years = range(start_date.year, end_date.year + 1)
        pool.starmap(run_backtest_for_year, [(year, interval, start_date, end_date) for year in years])

if __name__ == "__main__":
    run_parallel_backtests()
