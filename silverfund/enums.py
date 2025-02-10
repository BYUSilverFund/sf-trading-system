from enum import Enum


class Interval(Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class Weighting(Enum):
    EQUAL = "EQUAL"
    MARKET_CAP = "MARKET_CAP"
