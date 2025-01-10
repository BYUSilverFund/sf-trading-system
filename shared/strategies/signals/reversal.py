import polars as pl


def reversal_signal(chunk: pl.DataFrame, interval: str = "daily"):
    chunk = chunk.drop_nulls()

    # Set window size
    match interval:
        case "daily":
            window = 22
        case "monthly":
            window = 1

    # Signal transformation
    signal = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
        pl.col("logret")
        .rolling_sum(window_size=window, min_periods=window, center=False)
        .shift(1)  # Lag signal
        .over("permno")
        .alias("rev")
    )
    return signal
