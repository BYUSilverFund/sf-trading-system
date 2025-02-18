from datetime import date

import polars as pl


def check_columns(expected: list[str], actual: list[str]) -> None:
    left_unique = list(set(expected) - set(actual))
    right_unique = list(set(actual) - set(expected))

    if len(left_unique) > 0:
        raise ValueError(f"Columns missing: {left_unique}")

    if len(right_unique) > 0:
        raise ValueError(f"Extra columns found: {right_unique}")


def check_schema(expected: dict[str, pl.DataType], actual: pl.Schema) -> None:
    for col, dtype in expected.items():
        if actual[col] != dtype:
            raise ValueError(f"Column {col} has incorrect type: {actual[col]}, expected: {dtype}")


class Signal(pl.DataFrame):
    def __init__(self, signals: pl.DataFrame, signal_name: str) -> None:
        expected_order = ["date", "barrid", signal_name]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            signal_name: pl.Float64,
        }

        # Check columns
        check_columns(expected_order, signals.columns)

        # Check schema
        check_schema(valid_schema, signals.schema)

        # Reorder columns
        signals = signals.select(expected_order)

        # Initialize
        super().__init__(signals)


class Score(pl.DataFrame):
    def __init__(self, scores: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "score"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "score": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, scores.columns)

        # Check schema
        check_schema(valid_schema, scores.schema)

        # Reorder columns
        scores = scores.select(expected_order)

        # Initialize
        super().__init__(scores)


class Alpha(pl.DataFrame):
    def __init__(self, alphas: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "alpha"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "alpha": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, alphas.columns)

        # Check schema
        check_schema(valid_schema, alphas.schema)

        # Reorder columns
        alphas = alphas.select(expected_order)

        # Initialize
        super().__init__(alphas)

    def to_vector(self):
        return self.select("alpha").to_numpy()


class CovarianceMatrix(pl.DataFrame):
    def __init__(self, cov_mat: pl.DataFrame, barrids: list[str]) -> None:
        expected_order = ["barrid"] + sorted(barrids)

        valid_schema = {barrid: pl.Float64 for barrid in barrids}

        # Check columns
        check_columns(expected_order, cov_mat.columns)

        # Check schema
        check_schema(valid_schema, cov_mat.schema)

        # Reorder columns
        cov_mat = cov_mat.select(expected_order)

        # Initialize
        super().__init__(cov_mat)

    def to_matrix(self):
        return self.drop("barrid").to_numpy()


class Portfolio(pl.DataFrame):
    def __init__(self, portfolios: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "weight"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "weight": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, portfolios.columns)

        # Check schema
        check_schema(valid_schema, portfolios.schema)

        # Reorder columns
        portfolios = portfolios.select(expected_order)

        # Initialize
        super().__init__(portfolios)


class AssetReturns(pl.DataFrame):
    def __init__(self, returns: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "weight", "fwd_ret"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "weight": pl.Float64,
            "fwd_ret": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, returns.columns)

        # Check schema
        check_schema(valid_schema, returns.schema)

        # Reorder columns
        returns = returns.select(expected_order)

        # Initialize
        super().__init__(returns)
