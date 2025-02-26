from enum import Enum


class Interval(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


class Compounding(Enum):
    SUM = "sum"
    PRODUCT = "product"
