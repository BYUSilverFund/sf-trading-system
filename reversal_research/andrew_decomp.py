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
    returns=pl.read_parquet("barra_reversal_full_sample2.parquet").with_columns(
        pl.col("fwd_ret") / 100
    )
)


# Performance instance
performance = Performance(
    start_date=date(1995, 7, 31),
    end_date=date(2024, 12, 31),
    interval=Interval.MONTHLY,
    asset_returns=asset_returns,
    annualize=True,
)

# Chart
# title = "Monthly Classic Reversal Backtest"
title = "Monthly Barra Residual Reversal Backtest"
file_path = "/home/bwaits/Research/sf-trading-system/reversal_research/barra_backtest.png"
performance.plot_returns(
    compounding=Compounding.SUM, title=title, decompose=True, save_file_path=file_path
)

# Print summary
print(performance.summary())
