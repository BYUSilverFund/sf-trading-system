from datetime import date
from typing import Protocol

import cvxpy as cp
import polars as pl

import silverfund.data_access_layer as dal


class ConstraintConstructor(Protocol):
    def __call__(self, weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint: ...


def full_investment(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    return cp.sum(weights) == 1


def no_buying_on_margin(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    return weights <= 1


def long_only(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    return weights >= 0


def unit_beta(weights: cp.Variable, date_: date, barrids: list[str]) -> cp.Constraint:
    betas = (
        dal.load_total_risk(start_date=date_, end_date=date_)
        .filter(pl.col("barrid").is_in(barrids))
        .select(["date", "barrid", "predbeta"])
        .sort(["barrid", "date"])
    )

    betas = betas["predbeta"].to_list()
    return cp.sum(cp.multiply(weights, betas)) == 1
