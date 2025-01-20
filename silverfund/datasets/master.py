from datetime import date

import polars as pl
from tqdm import tqdm

from silverfund.datasets.barra_risk_forecasts import BarraRiskForecasts
from silverfund.datasets.crsp_daily import CRSPDaily
from silverfund.datasets.russell_constituents import RussellConstituents


class Master:

    def __init__(self, start_date: date, end_date: date, quiet: bool = True):
        self._start_date = start_date
        self._end_date = end_date or date.today()
        self._quiet = quiet

        # Load CRSP, mapping, and Barra Risk
        crsp = self._crsp()
        mapping = self._mapping()
        barra = self._barra_risk()

        if not quiet:
            print("Joining CRSP -> mapping = Master")
        self.df = crsp.join(mapping, on="permno", how="inner")

        if not quiet:
            print("Joining Master -> Barra Risk = Master")
        self.df = self.df.join(barra, on=["barrid", "date"], how="inner")

        # Reorder columns
        ids = ["date", "permno", "barrid", "ticker"]
        columns = ids + [col for col in self.df.columns if col not in ids]
        self.df = self.df.select(columns)

        # Sort
        self.df = self.df.sort(by=["permno", "date"])

    def load_all(self):
        return self.df

    def _crsp(self) -> pl.DataFrame:
        dataset = CRSPDaily(start_date=self._start_date, end_date=self._end_date)

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        crsp = []
        if self._quiet:
            for year in years:
                crsp.append(dataset.load(year))
        else:
            for year in tqdm(years, desc="Loading CRSP Daily Data"):
                crsp.append(dataset.load(year))

        crsp = pl.concat(crsp)

        # Date and security filters
        crsp = crsp.filter(
            pl.col("date").is_between(self._start_date, self._end_date),
            pl.col("shrcd").is_between(10, 11, closed="both"),  # Stocks
            pl.col("exchcd").is_between(1, 3, closed="both"),  # NYSE, NASDAQ, AMEX
        )

        return crsp

    def _barra_risk(self) -> pl.DataFrame:

        # Join all yearly datasets
        years = range(self._start_date.year, self._end_date.year + 1)

        barra_risk = []
        if self._quiet:
            for year in years:
                barra_risk.append(BarraRiskForecasts().load(year))
        else:
            for year in tqdm(years, desc="Loading Barra Risk Forecasts"):
                barra_risk.append(BarraRiskForecasts().load(year))

        barra_risk = pl.concat(barra_risk)

        return barra_risk

    def _mapping(self) -> pl.DataFrame:
        mapping = RussellConstituents().load_all()

        # Create mapping
        mapping = mapping.select(["permno", "barrid"]).unique().drop_nulls()

        # Cast permno to int
        mapping = mapping.with_columns(pl.col("permno").cast(pl.Int64))

        return mapping
