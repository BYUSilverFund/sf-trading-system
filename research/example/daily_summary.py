import polars as pl

from silverfund.enums import Compounding, Interval, Turnover
from silverfund.performance import Performance
from silverfund.records import AssetReturns

if __name__ == "__main__":
    folder = "research/example/results/"

    # Asset returns (result of backtester)
    data_file_path = folder + "daily_backtest.parquet"
    asset_returns = AssetReturns(returns=pl.read_parquet(data_file_path))

    # Performance instance
    performance = Performance(
        interval=Interval.DAILY,
        asset_returns=asset_returns,
        annualize=True,
    )

    # Chart
    # Decomposition
    title = "Example Daily Backtest"
    decomposed_plot_file_path = folder + "daily_backtest.png"
    performance.plot_returns(
        compounding=Compounding.SUM,
        title=title,
        decompose=True,
        save_file_path=decomposed_plot_file_path,
    )

    # Leverage
    title = "Example Daily Leverage"
    decomposed_plot_file_path = folder + "daily_backtest_lev.png"
    performance.plot_leverage(
        title=title,
        save_file_path=decomposed_plot_file_path,
    )

    # Turnover
    title = "Example Daily Leverage"
    decomposed_plot_file_path = folder + "daily_backtest_turn.png"
    performance.plot_two_sided_turnover(
        turnover=Turnover.RELATIVE,
        title=title,
        save_file_path=decomposed_plot_file_path,
    )

    # Table
    summary_file_path = folder + "daily_backtest.txt"
    performance.summary(summary_file_path)

    # Print summary
    print(performance.summary())
