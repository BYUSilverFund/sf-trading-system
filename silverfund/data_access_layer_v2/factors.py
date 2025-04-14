from datetime import date
from typing import Optional

import polars as pl

from silverfund.data_access_layer_v2.schema.factors import all_factors, industry_factors, risk_factors
from silverfund.data_access_layer_v2.schema.tables import factors_table
from silverfund.enums import FactorGroup


def load(start_date: date, end_date: date, factor_group: Optional[FactorGroup] = None) -> pl.DataFrame:
    factor_group = factor_group or FactorGroup.ALL

    match (factor_group):
        case FactorGroup.RISK:
            columns = risk_factors

        case FactorGroup.INDUSTRY:
            columns = industry_factors

        case FactorGroup.ALL:
            columns = all_factors

        case _:
            raise ValueError(f"Invalid factor group: {factor_group}")

    return factors_table.scan().filter(pl.col("date").is_between(start_date, end_date)).sort("date").select("date", *columns).collect()


def get_columns() -> str:
    return factors_table.columns()


if __name__ == "__main__":
    print(load(date(1975, 1, 1), date.today(), FactorGroup.RISK))
