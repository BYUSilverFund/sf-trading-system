from datetime import date

import polars as pl
from tqdm import tqdm

from silverfund.datasets.barra_returns import BarraReturns
from silverfund.datasets.barra_risk_forecasts import BarraRiskForecasts
from silverfund.datasets.barra_specific_returns import BarraSpecificReturns
from silverfund.datasets.trading_days import TradingDays
from silverfund.datasets.universe import Universe
from silverfund.enums import Interval


class MasterMonthly:

    def __init__(self, start_date: date, end_date: date, quiet: bool = True):
        self._start_date = start_date
        self._end_date = end_date or date.today()
        self._quiet = quiet

        # Load universe, returns, and risk
        universe = self._universe()
        trading_days = self._trading_days()
        barra_returns = self._barra_returns()
        barra_risk = self._barra_risk()
        barra_specific_returns = self._barra_specific_returns()

        # Merge 0
        if not quiet:
            print("Joining Universe + Trading Days = Master")
        self.df = universe.join(trading_days, on=["date"], how="left")

        # Merge 1
        if not quiet:
            print("Joining Master + Barra Returns = Master")
        self.df = self.df.join(barra_returns, on=["barrid", "date"], how="left")

        # Merge 2
        if not quiet:
            print("Joining Master + Barra Risk = Master")
        self.df = self.df.join(barra_risk, on=["barrid", "date"], how="left")

        # Merge 3
        if not quiet:
            print("Joining Master + Barra Specific Returns = Master")
        self.df = self.df.join(barra_specific_returns, on=["barrid", "date"], how="left")

        # Clean
        self.df = self._clean_merged(self.df)

        # Sort
        self.df = self.df.sort(by=["barrid", "date"])

    def load_all(self):
        return self.df

    def _universe(self):
        # Load
        universe = Universe().load()

        return universe

    def _trading_days(self):
        # Load
        trading_days = TradingDays(interval=Interval.MONTHLY).load_all()

        return trading_days

    def _barra_returns(self) -> pl.DataFrame:
        dataset = BarraReturns()

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        dfs = []
        if self._quiet:
            for year in years:
                # Load
                df = dataset.load(year)

                # Clean
                df = self._clean_barra_returns(df)

                dfs.append(df)

        else:
            for year in tqdm(years, desc="Loading Barra Returns"):
                # Load
                df = dataset.load(year)

                # Clean
                df = self._clean_barra_returns(df)

                dfs.append(df)

        dfs = pl.concat(dfs)

        # Date and security filters
        dfs = dfs.filter(pl.col("date").is_between(self._start_date, self._end_date))

        return dfs

    @staticmethod
    def _clean_barra_returns(df: pl.DataFrame) -> pl.DataFrame:
        # Add logret column
        df = df.with_columns(pl.col("ret").log1p().alias("logret"))

        # Add month column
        df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).sort(
            ["barrid", "date"]
        )

        df = df.group_by(["month", "barrid"]).agg(
            pl.col("date").last(),
            pl.col("currency").last(),
            pl.col("mktcap").last(),
            pl.col("price").last(),
            pl.col("logret").sum(),
        )

        # Compound up log returns
        df = df.with_columns((pl.col("logret").exp() - 1).alias("ret"))

        # Drop month and sort
        df = df.drop("month").sort(["barrid", "date"])

        return df

    def _barra_risk(self) -> pl.DataFrame:
        dataset = BarraRiskForecasts()

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        dfs = []
        if self._quiet:
            for year in years:
                # Load
                df = dataset.load(year)

                # Clean
                df = self._clean_barra_risk(df)

                dfs.append(df)
        else:
            for year in tqdm(years, desc="Loading Barra Risk Forecasts"):
                # Load
                df = dataset.load(year)

                # Clean
                df = self._clean_barra_risk(df)

                dfs.append(df)

        dfs = pl.concat(dfs)

        # Filter
        dfs = dfs.filter(pl.col("date").is_between(self._start_date, self._end_date))

        return dfs

    @staticmethod
    def _clean_barra_risk(df: pl.DataFrame) -> pl.DataFrame:
        # Add month column
        df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).sort(
            ["barrid", "date"]
        )

        df = df.group_by(["month", "barrid"]).agg(
            pl.col("date").last(),
            pl.col("div_yield").last(),
            pl.col("total_risk").last(),
            pl.col("spec_risk").last(),
            pl.col("histbeta").last(),
            pl.col("predbeta").last(),
        )

        # Drop month and sort
        df = df.drop("month").sort(["barrid", "date"])

        return df

    def _barra_specific_returns(self) -> pl.DataFrame:
        dataset = BarraSpecificReturns()

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        dfs = []
        if self._quiet:
            for year in years:
                # Load
                df = dataset.load(year)

                # Clean
                df = self._clean_barra_specific_returns(df)

                dfs.append(df)
        else:
            for year in tqdm(years, desc="Loading Barra Specific Returns"):
                # Load
                df = dataset.load(year)

                # Clean
                df = self._clean_barra_specific_returns(df)

                dfs.append(df)

        dfs = pl.concat(dfs)

        # Filter
        dfs = dfs.filter(pl.col("date").is_between(self._start_date, self._end_date))

        return dfs

    @staticmethod
    def _clean_barra_specific_returns(df: pl.DataFrame) -> pl.DataFrame:
        # Add log_spec_ret column
        df = df.with_columns(pl.col("spec_ret").log1p().alias("log_spec_ret"))

        # Add month column
        df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).sort(
            ["barrid", "date"]
        )

        df = df.group_by(["month", "barrid"]).agg(
            pl.col("date").last(),
            pl.col("log_spec_ret").sum(),
        )

        # Compound up log specific returns
        df = df.with_columns((pl.col("log_spec_ret").exp() - 1).alias("spec_ret"))

        # Drop month and sort
        df = df.drop("month").sort(["barrid", "date"])

        return df

    def _clean_merged(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.filter(pl.col("date").is_between(self._start_date, self._end_date))

        # Fill null values
        df = df.with_columns(
            pl.col("predbeta").fill_null(strategy="forward").over("barrid"),
            pl.col("total_risk").fill_null(strategy="forward").over("barrid"),
        )

        return df
