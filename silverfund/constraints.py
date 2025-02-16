from datetime import date
from typing import Protocol

import cvxpy as cp
import polars as pl

import silverfund.data_access_layer as dal
from silverfund.enums import Interval


class ConstraintConstructor(Protocol):
    def __call__(self, weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint: ...


def full_investment(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    return cp.sum(weights) == 1


def no_buying_on_margin(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    return weights <= 1


def long_only(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    return weights >= 0


def unit_beta(
    weights: cp.Variable, date_: date, barrids: list[str], interval: Interval
) -> cp.Constraint:
    # Cast to polars dataframe
    barrids_df = pl.DataFrame({"barrid": barrids})

    # Create betas dataframe
    betas_df = dal.load_total_risk(interval=interval, start_date=date_, end_date=date_).select(
        ["barrid", "predbeta"]
    )

    # Filter on universe, fill null with mean, and cast to np vector
    betas = (
        barrids_df.join(betas_df, how="left", on="barrid")
        .fill_null(strategy="mean")["predbeta"]
        .to_list()
    )

    return cp.sum(cp.multiply(weights, betas)) == 1
