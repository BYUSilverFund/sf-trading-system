from datetime import date

import gdown
import polars as pl

from silverfund.database import Database
from silverfund.datasets.dataset import Dataset


class CRSPMonthly(Dataset):
    """
    Monthly dataset for CRSP. This class handles the downloading, and cleaning in order to improve the reproducibility of our research.
    """

    def __init__(
        self,
        start_date: date,
        end_date: date | None = None,
        interval: str = "daily",
    ) -> None:
        """Initialize the CRSP Monthly dataset."""
        super().__init__(start_date, end_date, interval)

        self.db = Database()

    def download(self):
        if not self.db.exists(f"{self.table_name}_STG"):
            print("DOWNLOADING RAW FILE")

            file_id = "15E7hEZdUf9nVVzlZokPVkanglF4j_Vni"
            url = f"https://drive.google.com/uc?id={file_id}"

            raw_file_path = "raw_crsp_monthly.csv"
            gdown.download(url, raw_file_path, quiet=False)

            schema_overrides = {
                "NCUSIP": pl.Utf8,
                "CFACSHR": pl.Float32,
                "CFACPR": pl.Float32,
                "CUSIP": pl.Utf8,
                "DLRETX": pl.Utf8,
                "DLRET": pl.Utf8,
                "SICCD": pl.Utf8,
                "HSICCD": pl.Utf8,
            }

            df = pl.read_csv(raw_file_path, schema_overrides=schema_overrides)
            self.db.create(f"{self.table_name}_STG", df)

        self._clean()

    def _clean(self):
        print("CLEANING RAW FILE")

        # Raw file
        df = self.db.read(f"{self.table_name}_STG")

        # Lowercase columns
        df = df.rename({x: x.lower() for x in df.columns})

        # Filters
        df = df.filter(
            ((pl.col("shrcd") >= 10) & (pl.col("shrcd") <= 11))
            & ((pl.col("exchcd") >= 1) & (pl.col("exchcd") <= 3))
        )

        # Keep only necessary columns
        keep_columns = [
            "permno",
            "date",
            "cusip",
            "shrcd",
            "exchcd",
            "ticker",
            "shrout",
            "vol",
            "prc",
            "ret",
        ]
        df = df.select(keep_columns)

        # Fix ret and prc variables
        df = df.filter(pl.col("ret") != "C")  # Not sure what the C in the data represents (IPO?)
        df = df.with_columns(
            pl.col("prc")
            .abs()
            .alias("prc")  # Stocks with unavailable prc data are negated (bid-ask spread)
        )

        # Cast types
        df = df.with_columns(
            [
                pl.col("cusip").cast(pl.Utf8),  # Convert "cusip" to string
                pl.col("ret").cast(pl.Float64),  # Convert "ret" to numeric (float)
                pl.col("date").str.strptime(pl.Date),  # Convert "date" to datetime
            ]
        )

        # Sort values
        df = df.sort(by=["permno", "date"])

        self.db.create(self.core_table_name, df, overwrite=True)

    def load(self):
        data = self.db.read(self.core_table_name).filter(
            (pl.col("date") >= self.start_date) & (pl.col("date") <= self.end_date)
        )
        return data


if __name__ == "__main__":
    crsp = CRSPMonthly(start_date=date(2020, 1, 1), end_date=date(2024, 12, 31), interval="monthly")

    crsp.download()
    print(crsp.load())
