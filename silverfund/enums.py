from enum import Enum


class Interval(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


class Compounding(Enum):
    SUM = "sum"
    PRODUCT = "product"


class Turnover(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class FactorGroup(Enum):
    RISK = "risk"
    INDUSTRY = "industry"
    ALL = "all"
