from datetime import date

import silverfund.data_access_layer as dal
from silverfund.enums import Interval

df = dal.load_monthly_universe(
    start_date=date(1995, 7, 31),
    end_date=date(2024, 12, 31),
)

print(df)
