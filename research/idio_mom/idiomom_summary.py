from datetime import date

import polars as pl

from silverfund.enums import Compounding, Interval
from silverfund.performance import Performance
from silverfund.records import AssetReturns

folder = "/home/mcasella/sf-trading-system/research/idio_mom/"

# Asset returns (result of backtester)
data_file_path = folder + "idiomom_backtest_full_daily.parquet"
asset_returns = AssetReturns(returns=pl.read_parquet(data_file_path))

# Performance instance
performance = Performance(
    start_date=date(2004, 12, 31),
    end_date=date(2024, 3, 28),
    interval=Interval.DAILY,
    asset_returns=asset_returns,
    annualize=True,
)

# Chart
title = "Monthly Momentum Backtest"
decomposed_plot_file_path = folder + "monthly_momentum.png"
performance.plot_returns(
    compounding=Compounding.SUM,
    title=title,
    decompose=True,
    save_file_path=decomposed_plot_file_path,
)

# Table
summary_file_path = folder + "monthly_momentum2.txt"
performance.summary(summary_file_path)

# Print summary
print(performance.summary())
