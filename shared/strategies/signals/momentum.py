import polars as pl


def momentum_signal(chunk: pl.DataFrame, interval: str = "daily"):
    chunk = chunk.drop_nulls()

    # Set window size
    match interval:
        case "daily":
            window = 230
        case "monthly":
            window = 11

    # Signal transformation
    signal = chunk.with_columns(pl.col("ret").log1p().alias("logret")).with_columns(
        pl.col("logret")
        .rolling_sum(window_size=window, min_periods=window, center=False)
        .shift(1)  # Lag signal
        .over("ticker")
        .alias("mom")
    )
    return signal
