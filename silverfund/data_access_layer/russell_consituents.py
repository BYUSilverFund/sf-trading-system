import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def load_russell_constituents() -> pl.DataFrame:

    # File path
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    file_path = root_dir / "groups" / "grp_quant" / "data" / "russell_history.parquet"

    # Load
    df = pl.read_parquet(file_path)

    # Clean
    df = df.with_columns(
        pl.col("date").dt.date(),
        pl.col("obsdate").dt.date(),
        pl.col("enddate").dt.date(),
    )

    return df
