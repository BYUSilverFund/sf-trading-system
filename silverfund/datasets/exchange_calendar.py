from datetime import date
from typing import Optional

import exchange_calendars as xcals
import pandas as pd
import polars as pl


class ExchangeCalendar:

    def __init__(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        self._start_date = pd.Timestamp(start_date) if start_date else pd.Timestamp(2006, 1, 1, 0)  # min date in exchange calendars package
        self._end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()

        # Load calendar
        nyse = xcals.get_calendar("XNYS")
        schedule = nyse.sessions_in_range(self._start_date, self._end_date).to_list()
        schedule = pl.DataFrame(schedule).rename({"column_0": "date"}).with_columns(pl.col("date").dt.date())

        self.df = schedule

    def load(self) -> pl.DataFrame:
        return self.df


if __name__ == "__main__":
    cal = ExchangeCalendar(start_date=date(2007, 1, 1), end_date=date(2024, 1, 1)).load()

    print(cal)
