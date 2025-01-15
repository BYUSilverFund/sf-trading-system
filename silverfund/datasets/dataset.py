from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import polars as pl


class Dataset(ABC):
    """
    Abstract base class defining the interface for all dataset implementations.

    This interface standardizes how data is downloaded, loaded, and managed across
    different data sources (e.g., CRSP, COMPUSTAT, etc.).

    Attributes:
        start_date (date): Start date for the dataset
        end_date (date): End date for the dataset (defaults to today)
        interval (str): Data frequency (e.g., 'daily', 'monthly')
    """

    def __init__(
        self, start_date: date, end_date: Optional[date] = None, interval: str = "daily"
    ) -> None:
        """
        Initialize the dataset.

        Args:
            start_date: Starting date for data collection
            end_date: Ending date for data collection (defaults to today)
            interval: Data frequency ('daily', 'monthly', etc.)
        """
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.interval = interval.lower()

        # Validate interval
        if self.interval not in ["daily", "monthly"]:
            raise ValueError("interval must be either 'daily' or 'monthly'")

    @abstractmethod
    def download(self, redownload: bool = False) -> None:
        """
        Download data from the source and store it in the database.

        This method should handle:
        1. Checking if data already exists
        2. Downloading only missing data unless redownload is True
        3. Storing the data in a staging area
        4. Transforming the data into the standard format
        5. Merging with existing data

        Args:
            redownload: If True, force redownload of existing data
        """
        pass

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """
        Load data from the database for the specified date range.

        Returns:
            DataFrame containing the requested data with standardized columns:
                - date (Date): observation date
                - ticker (str): asset identifier
                - open (float): opening price
                - high (float): highest price
                - low (float): lowest price
                - close (float): closing price
                - volume (float): trading volume
                Additional columns may be included based on the data source.
        """
        pass

    @property
    def table_name(self) -> str:
        """Generate a standardized table name based on the dataset parameters."""
        start = self.start_date.strftime("%Y-%m-%d")
        end = self.end_date.strftime("%Y-%m-%d")
        return f"{self.__class__.__name__.upper()}_{self.interval.upper()}_{start}_{end}"

    @property
    def core_table_name(self) -> str:
        """Generate the name of the core table where all data is stored."""
        return f"{self.__class__.__name__.upper()}_{self.interval.upper()}"
