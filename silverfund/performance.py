from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
from tabulate import tabulate

import silverfund.data_access_layer as dal
from silverfund.enums import Compounding, Interval
from silverfund.records import AssetReturns


class Performance:
    def __init__(
        self,
        interval: Interval,
        start_date: date,
        end_date: date,
        asset_returns: AssetReturns,
        annualize: bool = True,
    ) -> None:
        # Set annualizing variables
        annual_scales = {Interval.DAILY: 252, Interval.MONTHLY: 12}

        self._annual_scale = annual_scales[interval]
        self._annualize = annualize

        # Load benchmark
        bmk = dal.load_benchmark(interval=interval, start_date=start_date, end_date=end_date)

        # Create total_ret, bmk_ret, and active_ret columns.
        self._asset_returns = (
            asset_returns.join(bmk, on=["date", "barrid"], how="left", suffix="_bmk")
            .with_columns((pl.col("weight") - pl.col("weight_bmk")).alias("weight_active"))
            .with_columns(
                (pl.col("weight") * pl.col("fwd_ret")).alias("total_ret"),
                (pl.col("weight_bmk") * pl.col("fwd_ret")).alias("bmk_ret"),
                (pl.col("weight_active") * pl.col("fwd_ret")).alias("active_ret"),
            )
        )

        # Create portfolio returns dataframe
        self._portfolio_returns = (
            self._asset_returns.group_by("date")
            .agg(pl.col("total_ret").sum(), pl.col("bmk_ret").sum(), pl.col("active_ret").sum())
            .sort("date")
        )

    def plot_returns(
        self, compounding: Compounding, decompose: bool = False, save_file_path: str | None = None
    ) -> None:
        df = self._portfolio_returns

        # Create cummulative returns dataframe
        # Sum of log returns
        if compounding == compounding.SUM:
            df = (
                df.with_columns(
                    pl.col("total_ret").log1p(),
                    pl.col("bmk_ret").log1p(),
                    pl.col("active_ret").log1p(),
                )
                .sort("date")
                .with_columns(
                    pl.col("total_ret").cum_sum(),
                    pl.col("bmk_ret").cum_sum(),
                    pl.col("active_ret").cum_sum(),
                )
            )

        # Product of gross returns
        if compounding == compounding.PRODUCT:
            df = (
                df.with_columns(
                    pl.col("total_ret") + 1,
                    pl.col("bmk_ret") + 1,
                    pl.col("active_ret") + 1,
                )
                .sort("date")
                .with_columns(
                    pl.col("total_ret").cum_prod(),
                    pl.col("bmk_ret").cum_prod(),
                    pl.col("active_ret").cum_prod(),
                )
                .with_columns(
                    pl.col("total_ret") - 1,
                    pl.col("bmk_ret") - 1,
                    pl.col("active_ret") - 1,
                )
            )

        # Put in percent space
        df = df.with_columns(
            pl.col("total_ret") * 100,
            pl.col("bmk_ret") * 100,
            pl.col("active_ret") * 100,
        )

        # Plot
        plt.figure(figsize=(10, 6))

        sns.lineplot(df, x="date", y="total_ret", label="Total", legend=None)

        if decompose:
            sns.lineplot(df, x="date", y="bmk_ret", label="Benchmark")
            sns.lineplot(df, x="date", y="active_ret", label="Active")
            plt.legend()

        plt.xlabel(None)
        plt.ylabel(f"Cummulative {compounding.value.title()} Returns (%)")
        plt.grid()

        # Save/show
        if save_file_path == None:
            plt.show()
        else:
            plt.savefig(save_file_path)

    @property
    def expected_return(self) -> float:
        result = self._portfolio_returns["total_ret"].mean()

        if self._annualize:
            result *= self._annual_scale

        return result

    @property
    def expected_benchmark_return(self) -> float:
        result = self._portfolio_returns["bmk_ret"].mean()

        if self._annualize:
            result *= self._annual_scale

        return result

    @property
    def expected_alpha(self) -> float:
        result = self._portfolio_returns["active_ret"].mean()

        if self._annualize:
            result *= self._annual_scale

        return result

    @property
    def volatility(self) -> float:
        result = self._portfolio_returns["total_ret"].std()

        if self._annualize:
            result *= np.sqrt(self._annual_scale)

        return result

    @property
    def benchmark_volatility(self) -> float:
        result = self._portfolio_returns["bmk_ret"].std()

        if self._annualize:
            result *= np.sqrt(self._annual_scale)

        return result

    @property
    def active_risk(self) -> float:
        result = self._portfolio_returns["active_ret"].std()

        if self._annualize:
            result *= np.sqrt(self._annual_scale)

        return result

    @property
    def sharpe_ratio(self) -> float:
        return self.expected_return / self.volatility

    @property
    def information_ratio(self) -> float:
        return self.expected_alpha / self.active_risk

    @property
    def tota_beta(self) -> float:
        formula = "total_ret ~ bmk_ret"
        result = smf.ols(formula, self._portfolio_returns).fit()
        return result.params["bmk_ret"]

    @property
    def tota_alpha(self) -> float:
        formula = "total_ret ~ bmk_ret"
        result = smf.ols(formula, self._portfolio_returns).fit()
        return result.params["Intercept"]

    def summary(self, save_file_path: str | None = None) -> str | None:
        # Create data rows for the table
        data = [
            [
                "Expected Return",
                f"{self.expected_return:.2%}",
                f"{self.expected_benchmark_return:.2%}",
                f"{self.expected_alpha:.2%}",
            ],
            [
                "Volatility",
                f"{self.volatility:.2%}",
                f"{self.benchmark_volatility:.2%}",
                f"{self.active_risk:.2%}",
            ],
            ["Sharpe Ratio", f"{self.sharpe_ratio:.2f}", "", ""],
            ["Information Ratio", f"{self.information_ratio:.2f}", "", ""],
            ["Total Beta", f"{self.tota_beta:.2f}", "", ""],
            ["Total Alpha", f"{self.tota_alpha:.2%}", "", ""],
        ]

        # Define headers
        headers = ["Metric", "Portfolio", "Benchmark", "Active"]

        # Generate the table
        table = tabulate(data, headers=headers, tablefmt="rounded_grid")
        title = "-" * 20 + " Performance Summary " + "-" * 20
        result = title + "\n\n" + table

        if save_file_path == None:
            return result
        else:
            with open(save_file_path, "w") as file:
                file.write(result)

    def __str__(self) -> str:
        return self.summary()


if __name__ == "__main__":
    data = pl.read_parquet("research/backtest_example.parquet")
    asset_returns = AssetReturns(data)

    p = Performance(
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        interval=Interval.MONTHLY,
        asset_returns=asset_returns,
    )

    print(p)
