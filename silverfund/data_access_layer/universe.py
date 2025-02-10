from datetime import date

import polars as pl

from silverfund.data_access_layer.russell_consituents import load_russell_constituents
from silverfund.data_access_layer.trading_days import load_trading_days
from silverfund.enums import Interval


def load_monthly_universe(start_date: date | None = None, end_date: date | None = None):
    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # Load trading days
    trading_days = load_trading_days(
        interval=Interval.MONTHLY,
        start_date=start_date,
        end_date=end_date,
    ).select("date")

    # Load russell constituents
    russell = load_russell_constituents()
    russell = russell.select(["date", "barrid"]).drop_nulls()

    # Create in_universe column
    russell = russell.with_columns(pl.lit(True).alias("in_universe"))

    # Pivot out and fill null
    russell = russell.pivot(on="barrid", index="date", values="in_universe").fill_null(False)

    # Merge trading days and russell constituents
    merged = trading_days.join(russell, on="date", how="left")

    # Forward fill
    merged = merged.fill_null(strategy="forward").drop_nulls()

    # Unpivot
    merged = merged.unpivot(value_name="in_universe", variable_name="barrid", index="date")

    # Keep in universe
    merged = merged.filter(pl.col("in_universe"))

    # Drop in_universe column
    merged = merged.drop("in_universe")

    # Filter
    merged = merged.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    merged = merged.sort(["barrid", "date"])

    return merged
