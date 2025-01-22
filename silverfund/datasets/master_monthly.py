from datetime import date

import polars as pl
from tqdm import tqdm

from silverfund.datasets.barra_returns import BarraReturns
from silverfund.datasets.barra_risk_forecasts import BarraRiskForecasts
from silverfund.datasets.russell_constituents import RussellConstituents


class MasterMonthly:

    def __init__(self, start_date: date, end_date: date, quiet: bool = True):
        self._start_date = start_date
        self._end_date = end_date or date.today()
        self._quiet = quiet

        # Load universe, returns, and risk
        russell = self._russell()
        barra_returns = self._barra_returns()
        barra_risk = self._barra_risk()

        # Merge 1
        if not quiet:
            print("Joining Russell + Barra Returns = Master")
        self.df = russell.join(barra_returns, on=["barrid", "date"], how="inner")

        # Merge 2
        if not quiet:
            print("Joining Master + Barra Risk = Master")
        self.df = self.df.join(barra_risk, on=["barrid", "date"], how="inner")

        # Sort
        self.df = self.df.sort(by=["barrid", "date"])

    def load_all(self):
        return self.df

    def _russell(self):
        # Load
        russell = RussellConstituents().load_all()

        # Select index columns
        russell = russell.select(["date", "barrid"]).unique()

        # Drop null barrids
        russell = russell.drop_nulls(subset=["barrid"])

        # Sort
        russell = russell.sort(["date", "barrid"])

        return russell

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
        df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).sort(["barrid", "date"])

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
        df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).sort(["barrid", "date"])

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


if __name__ == "__main__":
    master = MasterMonthly(start_date=date(2017, 1, 1), end_date=date(2017, 12, 31), quiet=False).load_all()

    # print(master)
    print(master.filter(pl.col("date").is_between(date(2017, 10, 1), date(2017, 10, 31))))
