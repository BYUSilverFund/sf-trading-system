import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl
from dotenv import load_dotenv


class Database:
    """
    A simple file-based database implementation using Polars DataFrames and Parquet files.

    This class provides basic database operations like create, read, insert, update, and delete
    while storing data in Parquet format. It also includes archiving capabilities and table
    existence validation.

    Attributes:
        _tables_dir (str): Directory path where table files are stored
        _archive_dir (str): Directory path where archived tables are stored
    """

    def __init__(self):
        """
        Initialize the database by setting up required directories and loading environment variables.
        Expects ROOT_DIR in environment variables pointing to the base directory.

        Raises:
            ValueError: If ROOT_DIR environment variable is not set
        """
        load_dotenv()

        root = os.getenv("ROOT")
        if not root:
            raise ValueError("ROOT environment variable must be set")

        self._tables_dir = os.path.join(root, "silverfund/database/.tables")
        self._archive_dir = os.path.join(root, "silverfund/database/.archive")

        # Create necessary directories
        os.makedirs(self._tables_dir, exist_ok=True)
        os.makedirs(self._archive_dir, exist_ok=True)

    def create(
        self,
        table_name: str,
        data: Optional[pl.DataFrame] = None,
        overwrite: bool = False,
        schema: Optional[Dict[str, pl.DataType]] = None,
    ) -> None:
        """
        Create a new table in the database.

        Args:
            table_name: Name of the table to create
            data: Optional DataFrame containing initial data
            overwrite: If True, overwrites existing table
            schema: Optional schema definition for empty tables

        Raises:
            ValueError: If table exists and overwrite is False
            TypeError: If schema is provided but not valid
        """
        table_path = self.get_table_path(table_name)

        if self.exists(table_name) and not overwrite:
            raise ValueError(f"Table '{table_name}' already exists and overwrite=False")

        df_to_write = None

        if data is not None:
            df_to_write = data
        elif schema:
            # Create empty DataFrame with specified schema
            df_to_write = pl.DataFrame(schema=schema)
        else:
            df_to_write = pl.DataFrame()

        df_to_write.write_parquet(table_path)

    def read(self, table_name: str, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Read data from a table.

        Args:
            table_name: Name of the table to read
            columns: Optional list of columns to read

        Returns:
            DataFrame containing the table data

        Raises:
            FileNotFoundError: If table doesn't exist
        """
        if not self.exists(table_name):
            raise FileNotFoundError(f"Table '{table_name}' does not exist")

        table_path = self.get_table_path(table_name)
        return pl.read_parquet(table_path, columns=columns)

    def insert(self, table_name: str, rows: pl.DataFrame, validate_schema: bool = True) -> None:
        """
        Insert new rows into a table.

        Args:
            table_name: Name of the target table
            rows: DataFrame containing rows to insert
            validate_schema: If True, validates schema compatibility

        Raises:
            ValueError: If schemas don't match and validate_schema=True
        """
        table = self.read(table_name)

        if validate_schema and table.schema != rows.schema:
            raise ValueError("Schema mismatch between table and new rows")

        table = pl.concat([table, rows])
        self.create(table_name, table, overwrite=True)

    def archive(self, table_name: str) -> None:
        """
        Move a table to the archive directory.

        Args:
            table_name: Name of the table to archive

        Raises:
            FileNotFoundError: If table doesn't exist
        """
        if not self.exists(table_name):
            raise FileNotFoundError(f"Table '{table_name}' does not exist")

        src_table_path = self.get_table_path(table_name)
        dst_table_path = os.path.join(self._archive_dir, table_name)

        shutil.move(src_table_path, dst_table_path)

    def delete(self, table_name: str) -> None:
        """
        Delete a table from the database.

        Args:
            table_name: Name of the table to delete

        Raises:
            FileNotFoundError: If table doesn't exist
        """
        if not self.exists(table_name):
            raise FileNotFoundError(f"Table '{table_name}' does not exist")

        table_path = self.get_table_path(table_name)
        os.remove(table_path)

    def get_table_path(self, table_name: str) -> str:
        """Get the full file path for a table."""
        return os.path.join(self._tables_dir, f"{table_name}.parquet")

    def exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        table_path = self.get_table_path(table_name)
        return os.path.exists(table_path)

    def list_tables(self) -> List[str]:
        """Get a list of all tables in the database."""
        files = os.listdir(self._tables_dir)
        return [f.replace(".parquet", "") for f in files if f.endswith(".parquet")]
