import os
from datetime import datetime

import numpy as np
import pandas as pd

from silverfund.trader.api.margin_impact import calc_margin_impact
from silverfund.trader.api.portfolio_info import get_portfolio_holdings
from silverfund.trader.api.security_info import get_security_info
from silverfund.trader.api.trader import *
from silverfund.trader.utils.rebalancer import rebalance_port
from silverfund.trader.utils.validation import *


def main():
    """
    Rebalance the portfolio in TWS to match the optimal weights provided in the input DataFrame

    Calls PortfolioHoldings to get current positions, calls LastClosePrice to get the last close
    price for each security, calculates the needed change in position based on the current
    positions and their closing prices, and places orders in TWS for each needed change.
    """

    day, month, year = get_timestamp()

    # # # INSERT FILE PATH OF CSV FILE WITH OPTIMAL WEIGHTS IN THE BELOW LINE # # #
    optimal_wts_df = pd.read_csv("optimal_weights_{year}-{month:02d}-{day:02d}.csv")

    optimal_wts_df["ticker"] = [
        " ".join(ticker.split(".")) for ticker in optimal_wts_df["ticker"].values
    ]

    # Validate and clean the input data
    validate_input(optimal_wts_df)

    print("\nBeginning rebalance:\n")
    optimal_wts_df = optimal_wts_df.set_index("ticker")

    # Get current portfolio holdings
    holdings_df = get_portfolio_holdings()

    # Validate and retrieve the holding prices for new stocks
    last_price_df = validate_pricing_of_new_stocks(holdings_df, optimal_wts_df)

    ### Compute the target position (quantity) for each security of the optimal wts portfolio ###
    mkt_val = holdings_df["marketValue"].sum()
    # Validate weights sum to 1
    wts_sum = validate_weight_sums(optimal_wts_df)
    optimal_wts_df["wts"] /= wts_sum
    optimal_wts_df["marketValue"] = mkt_val * optimal_wts_df["wts"]  # target dollar value
    optimal_wts_df = pd.merge(
        optimal_wts_df, holdings_df[["marketPrice"]], how="outer", left_index=True, right_index=True
    )
    print(optimal_wts_df["marketPrice"])
    for ticker in optimal_wts_df[optimal_wts_df.isna().any(axis=1)].index:
        if ticker in last_price_df.index:
            optimal_wts_df.loc[ticker, "marketPrice"] = last_price_df.loc[ticker, "price"]
        else:
            optimal_wts_df.drop(ticker)

    optimal_wts_df["quantity"] = optimal_wts_df["marketValue"] / optimal_wts_df["marketPrice"]

    ## Andrew's optimizer is used here
    optimal_wts_df["quantity"] = np.floor(optimal_wts_df["quantity"])  # round down to integer

    ### Construct an order DataFrame by merging optimal_wts_df and holdings_df ###
    order_df = optimal_wts_df[["wts", "quantity", "marketPrice"]]
    order_df = pd.merge(
        order_df, holdings_df[["position"]], how="outer", left_index=True, right_index=True
    ).fillna(0)
    order_df = order_df.drop("$USD")  # drop cash as a position

    ### Initialize an output DataFrame to hold the weights and quantities of the rebalanced portfolio ###
    rebalanced_df = order_df[["wts", "quantity", "marketPrice"]]
    rebalanced_df["error"] = "No"  # default; will change in execution if an error occurs

    ### Compute the order quantities by subtracting the current position from the current position ###
    order_df["quantity"] = order_df["quantity"] - order_df["position"]
    order_df.to_csv("intermediate.csv")
    rebalanced_df["change"] = order_df["quantity"]  # store change from previous portfolio in output
    order_df = order_df.drop(["wts", "position"], axis=1)  # no longer needed for order

    ### Trim the order data frame to include nonzero order requests ###
    zero_tolerance = 0.0001
    order_df = order_df[order_df["quantity"].abs() > zero_tolerance]
    order_df = pd.merge(
        order_df, get_security_info(order_df.index), how="inner", left_index=True, right_index=True
    )
    order_df = calc_margin_impact(order_df)

    ### Create an instance of the Trader class, connect to TWS, and run the app ###
    app = Trader(order_df, error_list)
    app.connect("127.0.0.1", 7496, 9)
    app.run()
    # Record any errors that occurred during the order process in the output DataFrame
    # Initialize an empty list to hold tickers whose orders have errors in execution
    error_list = []
    for ticker in error_list:
        rebalanced_df.loc[ticker, "error"] = "Yes"

    ### Save the new positions as well as order info for reference ###
    rebalanced_df = rebalanced_df[["quantity", "wts", "change", "marketPrice", "error"]]
    rebalanced_df.to_csv("rebalanced.csv")

    ### Inform the user of any errors and the completion of the rebalance. ###
    print("\nCompleted submission of rebalance orders.")
    if error_list:
        print("\nOrders for the following securities returned errors:\n")
        print(error_list)
        print()
    print("Review TWS for order status.\n")


if __name__ == "__main__":
    main()


def get_timestamp():
    current_utc_time = datetime.now(datetime.timezone.utc)
    return current_utc_time.day, current_utc_time.month, current_utc_time.year


# Query Fills

# Dividend Query - shows up in NAV, rate charged? Funds cleared for trading, T+1 settlement
# Solution: use Current Available Funds - Total Cash + Settled Cash
# Note: Not sure if settled cash is available in API...

# Query borrow availability/cost
# Transaction cost, holding cost (financing cost + borrow cost), alpha,
