from datetime import datetime, timezone

import pandas as pd

from silverfund.trader.api.last_price import get_last_price


def portfolio():
    barra_weights = pd.read_csv("silverfund/trader/data/current_portfolio.csv")
    mapping_df = pd.read_csv("silverfund/trader/data/mega_monthly.csv")

    barra_weights["weight"] = barra_weights["weight"].round(4)
    barra_weights = barra_weights[barra_weights["weight"] > 0]

    print("Total weight after filtering:", barra_weights["weight"].sum())

    mapping_df = mapping_df[["barrid", "ticker_crsp"]]
    mapping_df = mapping_df.drop_duplicates(subset=["barrid"], keep="first")

    # Merge to map barrid to ticker_crsp
    df_merged = barra_weights.merge(mapping_df, on="barrid", how="left")
    df_merged = df_merged.rename(columns={"ticker_crsp": "ticker", "weight": "wts"})
    if df_merged["ticker"].isna().sum() > 0:
        print(f'Count not find ticker symbol for {df_merged["ticker"].isna()}')
        # add to basically
        df_merged = df_merged.dropna(subset=["ticker"])

    df_merged = df_merged[["ticker", "wts"]]

    last_price_df, new_tickers = get_last_price(df_merged["ticker"])

    # Make a list of all securities missing close price data, which must be dropped from the order
    to_drop = []
    for ticker in new_tickers:
        if ticker not in last_price_df.index:
            to_drop.append(ticker)
    if not to_drop:
        day, month, year = get_timestamp()
        df_merged.to_csv(
            f"silverfund/trader/data/optimal_weights_{year}-{month:02d}-{day:02d}.csv", index=False
        )

    return df_merged


def get_timestamp():
    current_utc_time = datetime.now(timezone.utc)
    return current_utc_time.day, current_utc_time.month, current_utc_time.year


print(portfolio().head())
