from datetime import date

import polars as pl
from tqdm import tqdm

import silverfund.data_access_layer as dal
from silverfund.alphas import Alpha
from silverfund.enums import Interval
from silverfund.strategies import Strategy


class ProfitAndLoss(pl.DataFrame):
    def __init__(self, pnl: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "weight", "fwd_ret"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "weight": pl.Float64,
            "fwd_ret": pl.Float64,
        }

        # Check if all required columns exist
        if set(expected_order) != set(pnl.columns):
            missing = set(expected_order) - set(pnl.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure correct column types
        for col, dtype in valid_schema.items():
            if pnl.schema[col] != dtype:
                raise ValueError(
                    f"Column {col} has incorrect type: {pnl.schema[col]}, expected: {dtype}"
                )

        # Reorder columns
        pnl = pnl.select(expected_order)

        # Initialize
        super().__init__(pnl)


class Backtester:
    def __init__(self, start_date: date, end_date: date, strategy: Strategy):
        self._start_date = start_date
        self._end_date = end_date
        self._strategy = strategy

    def run_sequential(self) -> ProfitAndLoss:
        # Universe, training_data, and testing_data will all be parameters in the future.
        universe = dal.load_monthly_universe(start_date=self._start_date, end_date=self._end_date)

        training_data = universe.join(
            dal.load_barra_returns(
                interval=Interval.MONTHLY, start_date=self._start_date, end_date=self._end_date
            ),
            on=["date", "barrid"],
            how="left",
        ).sort(["barrid", "date"])

        testing_data = (
            training_data.with_columns((pl.col("ret").shift(-1) * 100).alias("fwd_ret"))
            .select(["date", "barrid", "fwd_ret"])
            .sort(["barrid", "date"])
        )

        # Calculate signals, scores, and alphas
        signals = self._strategy.signal_constructor(training_data)
        scores = self._strategy.score_constructor(signals)
        alphas = self._strategy.alpha_constructor(scores)

        # Get unique periods
        periods = universe["date"].unique().sort().to_list()
        portfolios = []
        for period in tqdm(periods, desc="Computing portfolios"):

            # Get portfolio constructor parameters
            period_barrids = universe.filter(pl.col("date") == period)["barrid"].sort().to_list()
            period_alphas = Alpha(alphas.filter(pl.col("date") == period).sort(["barrid"]))

            # Construct period portfolio
            portfolio = self._strategy.portfolio_constructor(
                period=period,
                barrids=period_barrids,
                alphas=period_alphas,
                constraints=self._strategy.constraints,
            )
            portfolios.append(portfolio)

        # Concatenate portfolios
        portfolios = pl.concat(portfolios)

        # Join forward returns on portfolios
        pnl = portfolios.join(testing_data, on=["barrid", "date"], how="left")
        pnl = pnl.sort(["barrid", "date"])

        return ProfitAndLoss(pnl)
