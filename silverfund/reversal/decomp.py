from datetime import date

import polars as pl

from silverfund.enums import Compounding, Interval
from silverfund.performance import Performance
from silverfund.records import AssetReturns

# folder = "reversal_research/"

# Asset returns (result of backtester)
# data_file_path = folder + "barra_reversal_full_sample2.parquet"
# data_file_path = folder + "classic_reversal_full_sample.parquet"
# asset_returns = AssetReturns(returns=pl.read_parquet(data_file_path).with_columns(pl.col('fwd_ret') / 100))
# asset_returns = AssetReturns(returns=pl.read_parquet("classic_reversal_full_sample.parquet").with_columns(pl.col('fwd_ret') / 100))
asset_returns = AssetReturns(
    returns=pl.read_parquet("silverfund/reversal/daily_reversal_2_year_2.parquet").with_columns(
        pl.col("fwd_ret") / 100
    )
)


# Performance instance
performance = Performance(
    interval=Interval.DAILY,
    asset_returns=asset_returns,
    annualize=True,
)

# Chart
# title = "Monthly Classic Reversal Backtest"
title = "Daily Barra Residual Reversal Backtest"
file_path = "/home/joshhoag/projects/47fund_repo/silverfund/reversal/daily_barra_backtest.png"
performance.plot_returns(
    compounding=Compounding.SUM, title=title, decompose=True, save_file_path=file_path
)

# Print summary
print(performance.summary())
