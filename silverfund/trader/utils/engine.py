from silverfund.trader.api.security_info import get_security_info
from silverfund.trader.utils.validation import *


def compute_target_positions(holdings_df, optimal_wts_df, last_price_df):
    ### Compute the target position (quantity) for each security of the optimal wts portfolio ###
    mkt_val = holdings_df["marketValue"].sum()
    # Validate weights sum to 1
    wts_sum = validate_weight_sums(optimal_wts_df)
    optimal_wts_df["wts"] /= wts_sum
    optimal_wts_df["marketValue"] = mkt_val * optimal_wts_df["wts"]  # target dollar value
    optimal_wts_df = pd.merge(
        optimal_wts_df, holdings_df[["marketPrice"]], how="outer", left_index=True, right_index=True
    )
    for ticker in optimal_wts_df[optimal_wts_df.isna().any(axis=1)].index:
        if ticker in last_price_df.index:
            optimal_wts_df.loc[ticker, "marketPrice"] = last_price_df.loc[ticker, "price"]
        else:
            optimal_wts_df.drop(ticker, inplace=True)
    optimal_wts_df["quantity"] = optimal_wts_df["marketValue"] / optimal_wts_df["marketPrice"]
    return optimal_wts_df


def construct_order_form(order_df, holdings_df):
    ### Construct an order DataFrame by merging optimal_wts_df and holdings_df ###
    order_df = pd.merge(
        order_df, holdings_df[["position"]], how="outer", left_index=True, right_index=True
    ).fillna(0)
    order_df = order_df.drop("$USD")  # drop cash as a position
    return order_df


def compute_order_quantities(order_df, rebalanced_df):
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
    return order_df
