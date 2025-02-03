import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


class BarraRiskForecasts:

    def __init__(self) -> None:
        load_dotenv()

        parts = os.getenv("ROOT").split("/")
        home = parts[1]
        user = parts[2]
        root_dir = Path(f"/{home}/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow_asset"
        self._files = os.listdir(self._folder)

    def load(self, year: int) -> pl.DataFrame:

        file = f"asset_{year}.parquet"

        return self.clean(pl.read_parquet(self._folder / file))

    def get_all_years(self) -> list[int]:

        years = []
        for file in self._files:
            year = file.split("_")[1].split(".")[0]
            years.append(year)

        return years

    def get_total_vol_forcasts(self, date: date, barr_id_list: list[str] = None) -> pl.DataFrame:
        year = date.year

        risk_forecast_year = BarraRiskForecasts().load(year)
        if barr_id_list:
            return risk_forecast_year.filter(
                pl.col("Date") == date, pl.col("Barrid").is_in(barr_id_list)
            ).select(["Date", "Barrid", "total_risk"])
        else:
            return risk_forecast_year.filter(pl.col("Date") == date).select(
                ["Date", "Barrid", "total_risk"]
            )

    def get_spec_vol_forcasts(self, date: date, barr_id_list: list[str] = None) -> pl.DataFrame:
        year = date.year

        risk_forecast_year = BarraRiskForecasts().load(year)
        if barr_id_list:
            return risk_forecast_year.filter(
                pl.col("Date") == date, pl.col("Barrid").is_in(barr_id_list)
            ).select(["Date", "Barrid", "spec_risk"])
        else:
            return risk_forecast_year.filter(pl.col("Date") == date).select(
                ["Date", "Barrid", "spec_risk"]
            )

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:
        # Drop index column
        df = df.drop("__index_level_0__")

        # Cast and rename date
        df = df.with_columns(pl.col("DataDate").dt.date().alias("date")).drop("DataDate")

        # Lowercase columns
        df = df.rename({col: col.lower() for col in df.columns})

        # Reorder columns
        df = df.select(
            ["date", "barrid"] + [col for col in df.columns if col not in ["date", "barrid"]]
        )

        # Sort
        df = df.sort(by=["date", "barrid"])

        return df
