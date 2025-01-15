import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.database import Database
from silverfund.datasets.dataset import Dataset


class BarraFactorExposures:

    def __init__(self) -> None:
        self.db = Database()

        load_dotenv()

        user = os.getenv("ROOT").split("/")[2]
        root_dir = Path(f"/home/{user}")

        self._folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"
        self._files = os.listdir(self._folder)

    def download(self, redownload: bool = False):
        years = range(1995, 2026)

        dfs = []
        for year in tqdm(years, desc="Downloading parquet files"):
            df = pl.read_parquet(self._folder / f"exposures_{year}.parquet")
            dfs.append(self.clean(df))

        result = pl.concat(dfs)

        self.db.create("BARRA_FACTOR_EXPOSURES", result)

    def load(self) -> pl.DataFrame:

        return self.db.read("BARRA_FACTOR_EXPOSURES")

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:

        # Rename columns headers (cast to date)
        new_cols = list(map(lambda x: x.split(" ")[0], df.columns))
        df = df.rename({col: new_col for col, new_col in zip(df.columns, new_cols)})

        # Split Combined colum into Barrid and Factor
        df = (
            df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
            .with_columns(
                pl.col("parts").list.first().alias("Barrid"),
                pl.col("parts").list.last().alias("Factor"),
            )
            .drop(["Combined", "parts"])
        )

        # Melt date headers into a column
        df = df.unpivot(index=["Barrid", "Factor"], variable_name="Date", value_name="Value")

        # Sort
        df = df.sort(by=["Barrid", "Date"])

        return df


if __name__ == "__main__":
    bf = BarraFactorExposures()
    bf.download()
    print(bf.load())
