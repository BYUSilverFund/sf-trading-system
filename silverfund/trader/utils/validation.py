import pandas as pd

from silverfund.trader.api.last_price import get_last_price


def validate_input(df):
    # Check for invalid values in the input DataFrame.ssssssssssssss
    nan_df = df[df.isna().any(axis=1)]
    if len(nan_df) > 0:

        # One or more invalid value has been found. Present this result and ask to proceed.
        print(nan_df)
        print(f"\nThe DataFrame provided has {len(nan_df)} invalid entries listed above.")
        goOn = input("Remove these entries and proceed? Y/N ")

        if goOn != "Y" and goOn != "y":
            quit()  # End the program early if the user wishes not to drop the invalid entries

        else:
            # Otherwise, drop the invalid entries and continue
            df.dropna(inplace=True)


def validate_pricing_of_new_stocks(holdings_df, optimal_wts_df, account_type):

    new_tickers = [ticker for ticker in optimal_wts_df.index if ticker not in holdings_df.index]

    if new_tickers:

        last_price_df, new_tickers = get_last_price(new_tickers, account_type)

        # Make a list of all securities missing close price data, which must be dropped from the order
        to_drop = []
        for ticker in new_tickers:
            if ticker not in last_price_df.index:
                to_drop.append(ticker)

        if to_drop:  # Data could not be retrieved for at least one security

            # Inform the user of the issue and ask for permission to continue
            print("Securities that will not be included in this rebalancing: \n\n", to_drop, "\n")
            goOn = input(f"Proceed without including these {len(to_drop)} securities? Y/N ")
            print()

            if goOn != "Y" and goOn != "y":

                # End the program early if the user wishes not to drop these securities from the order
                quit()

            else:

                # Otherwise, drop the securities with missing data from the order
                optimal_wts_df = optimal_wts_df.drop(to_drop)

        else:  # Data was successfully retieved for all securities

            print("\nLast price retrieval successful.\n")
    else:
        last_price_df = pd.DataFrame()

    return last_price_df


def validate_weight_sums(df):
    # Ensure that the wts column of optimal_wts_df sums to 1.
    wts_sum = df["wts"].sum()

    if wts_sum <= 0.97 or wts_sum > 1:
        print(wts_sum)
        goOn = input("Weights do not sum between .97 and 1. Do you want to continue? Y/N \n")
        if goOn != "Y" and goOn != "y":
            # End the program early if the user wishes not to drop these securities from the order
            print("Ending early due to an improper weight sum")
            quit()
        print()

    return wts_sum
