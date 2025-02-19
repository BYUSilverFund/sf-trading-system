from datetime import date

import polars as pl

from silverfund.enums import Compounding, Interval
from silverfund.performance import Performance
from silverfund.records import AssetReturns


# Asset returns (result of backtester)
data_file_path = "lowBeta_backtest.parquet"
asset_returns = AssetReturns(returns=pl.read_parquet(data_file_path))

# Performance instance
performance = Performance(
    start_date=date(2018, 1, 1),
    end_date=date(2023, 12, 31),
    interval=Interval.MONTHLY,
    asset_returns=asset_returns,
    annualize=True,
)

# Chart
decomposed_plot_file_path = "lowBeta_decomposed.png"
performance.plot_returns(
    compounding=Compounding.SUM, decompose=True, save_file_path=decomposed_plot_file_path
)

# Table
summary_file_path = "lowBeta_summary.txt"
performance.summary(summary_file_path)

# Print summary
print(performance.summary())
