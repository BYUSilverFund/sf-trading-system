from datetime import date

import exchange_calendars as xcals
import polars as pl
from tqdm import tqdm

from silverfund.datasets.barra_returns import BarraReturns
from silverfund.datasets.barra_risk_forecasts import BarraRiskForecasts
from silverfund.datasets.exchange_calendar import ExchangeCalendar
from silverfund.datasets.universe import Universe


class Master:

    def __init__(self, start_date: date, end_date: date, quiet: bool = True):
        self._start_date = start_date
        self._end_date = end_date or date.today()
        self._quiet = quiet

        # Load universe, returns, and risk
        universe = self._universe()
        barra_returns = self._barra_returns()
        barra_risk = self._barra_risk()

        # Merge 1
        if not quiet:
            print("Joining Universe <- Barra Returns = Master")
        self.df = universe.join(barra_returns, on=["barrid", "date"], how="left")

        # Merge 1
        if not quiet:
            print("Joining Master <- Barra Risk = Master")
        self.df = self.df.join(barra_risk, on=["barrid", "date"], how="left")

        # Sort
        self.df = self.df.sort(by=["barrid", "date"])

    def load_all(self):
        return self.df

    def _universe(self):
        univ = Universe(start_date=self._start_date, end_date=self._end_date).load()

        return univ

    def _barra_returns(self) -> pl.DataFrame:
        dataset = BarraReturns()

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        dfs = []
        if self._quiet:
            for year in years:
                dfs.append(dataset.load(year))

        else:
            for year in tqdm(years, desc="Loading Barra Returns"):
                # Load
                df = dataset.load(year)

                # Reorder columns
                df = df.select(["date", "barrid", "price", "pricesource", "currency", "mktcap", "ret"])

                dfs.append(df)

        dfs = pl.concat(dfs)

        # Date and security filters
        dfs = dfs.filter(pl.col("date").is_between(self._start_date, self._end_date))

        return dfs

    def _barra_risk(self) -> pl.DataFrame:
        dataset = BarraRiskForecasts()

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        dfs = []
        if self._quiet:
            for year in years:
                dfs.append(dataset.load(year))
        else:
            for year in tqdm(years, desc="Loading Barra Risk Forecasts"):
                dfs.append(dataset.load(year))

        dfs = pl.concat(dfs)

        # Filter
        dfs = dfs.filter(pl.col("date").is_between(self._start_date, self._end_date))

        return dfs


if __name__ == "__main__":
    master = Master(start_date=date(2020, 1, 1), end_date=date.today(), quiet=False).load_all()

    print(master)
