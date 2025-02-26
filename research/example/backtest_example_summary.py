import polars as pl

from silverfund.enums import Compounding, Interval
from silverfund.performance import Performance
from silverfund.records import AssetReturns

if __name__ == "__main__":
    folder = "research/example/results/"

    # Asset returns (result of backtester)
    data_file_path = folder + "backtest_example.parquet"
    asset_returns = AssetReturns(returns=pl.read_parquet(data_file_path))

    # Performance instance
    performance = Performance(
        interval=Interval.MONTHLY,
        asset_returns=asset_returns,
        annualize=True,
    )

    # Chart
    title = "Example Monthly Backtest"
    decomposed_plot_file_path = folder + "backtest_example.png"
    performance.plot_returns(
        compounding=Compounding.SUM,
        title=title,
        decompose=True,
        save_file_path=decomposed_plot_file_path,
    )

    # Table
    summary_file_path = folder + "backtest_example.txt"
    performance.summary(summary_file_path)

    # Print summary
    print(performance.summary())
