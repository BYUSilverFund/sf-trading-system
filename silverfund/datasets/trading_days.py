from datetime import date

import polars as pl

from silverfund.components.enums import Interval
from silverfund.datasets.crsp_daily import CRSPDaily
from silverfund.datasets.crsp_monthly import CRSPMonthly


class TradingDays:

    def __init__(self, start_date: date, end_date: date, interval: Interval = Interval.MONTHLY) -> None:

        if interval == Interval.MONTHLY:

            crsp = CRSPMonthly(start_date=start_date, end_date=end_date).load_all()

        elif interval == Interval.DAILY:

            crsp = CRSPDaily(start_date=start_date, end_date=end_date).load_all()

        self.df = crsp.select("date").unique().sort("date")

    def load(self) -> pl.DataFrame:
        return self.df
