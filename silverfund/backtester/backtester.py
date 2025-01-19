import time
from datetime import date

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from silverfund.components.chunked_data import ChunkedData
from silverfund.components.enums import Interval
from silverfund.components.strategies.momentum_strategy import MomentumStrategy
from silverfund.components.strategies.strategy import Strategy
from silverfund.datasets import CRSPDaily, Master


class Backtester:

    def __init__(
        self,
        start_date: date,
        end_date: date,
        interval: Interval,
        historical_data: pl.DataFrame,
        strategy: Strategy,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.historical_data = historical_data
        self.strategy = strategy(interval)

    def run(self):

        # Create chunks
        chunked_data = ChunkedData(
            data=self.historical_data,
            interval=self.interval,
            window=self.strategy.window,
        )

        portfolios = chunked_data.apply_strategy(self.strategy)

        portfolios = pl.concat(portfolios)

        merged = self.historical_data.join(portfolios, how="inner", on=["date", "permno"])

        merged = merged.with_columns((pl.col("weight") * pl.col("ret")).alias("weighted_ret"))

        pnl = (
            merged.group_by("date")
            .agg(weighted_ret_mean=pl.col("weighted_ret").sum(), n_assets=pl.col("date").count())
            .sort(by=["date"])
        )

        pnl = (
            pnl.with_columns(pl.col("weighted_ret_mean").alias("portfolio_ret"))
            .with_columns(pl.col("portfolio_ret").log1p().alias("portfolio_logret"))
            .with_columns(
                ((pl.col("portfolio_ret") + 1).cum_prod() - 1).alias("cumprod"),
                pl.col("portfolio_logret").cum_sum().alias("cumsum"),
            )
        )

        # Output
        print("\n" + "-" * 50 + " Backtest P&L " + "-" * 50)

        print(pnl)

        sns.lineplot(data=pnl, x="date", y="cumsum")
        plt.ylabel("Cummulative returns (sum)")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    start_date = date(2006, 1, 1)
    end_date = date(2024, 12, 31)

    # Load historical dataset
    historical_data = (
        CRSPDaily(
            start_date=start_date,
            end_date=end_date,
        )
        .load_all()
        .select(["date", "permno", "ret"])
    )

    bt = Backtester(
        start_date=start_date,
        end_date=end_date,
        interval=Interval.MONTHLY,
        historical_data=historical_data,
        strategy=MomentumStrategy,
    )

    bt.run()
