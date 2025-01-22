from threading import Timer

import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class PortfolioInfoApp(EWrapper, EClient):
    """
    App to get account value and portfolio holdings.
    Internal use only.
    """

    def __init__(self, account_value_df, portfolio_df):

        self.account_value_df = account_value_df
        self.portfolio_df = portfolio_df
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):

        if errorCode in [2104, 2106, 2158]:
            return  # These codes simply indicate that the API made connection well. Do nothing.

        return super().error(errorCode, errorString, advancedOrderReject)

    def nextValidId(self, orderId):
        """Automatically executed whenever the app is run. Calls self.start."""

        self.start()

    def updatePortfolio(
        self,
        contract: Contract,
        position: float,
        marketPrice: float,
        marketValue: float,
        averageCost: float,
        unrealizedPNL: float,
        realizedPNL: float,
        accountName: str,
    ):
        """Saves information about the currently held position of securities"""

        self.portfolio_df.loc[len(self.portfolio_df)] = [
            contract.symbol,
            contract.secType,
            contract.exchange,
            position,
            marketPrice,
            marketValue,
            averageCost,
            unrealizedPNL,
            realizedPNL,
            accountName,
        ]

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """Saves the value of each account to the account value DataFrame."""

        self.account_value_df.loc[len(self.account_value_df)] = [key, val, currency, accountName]

    def accountDownloadEnd(self, accountName: str):
        """Automatically calls self.stop when the accout download process ends."""

        self.stop()

    def start(self):
        """Requests updates on portfolio positions and account values."""

        # Account number can be omitted when using reqAccountUpdates with single account structure
        self.reqAccountUpdates(True, "")

    def stop(self):
        """Cancels account requests and disconnects from TWS."""

        self.reqAccountUpdates(False, "")
        self.done = True
        self.disconnect()


def get_portfolio_holdings(port=7496):
    """Retrieves the current portfolio holdings and account value
    :return: DataFrame of portfolio holdings
    """
    # Initialize two DataFrames for receiving security positions and account values.
    account_value_df = pd.DataFrame(columns=["key", "value", "currency", "accountName"])
    portfolio_df = pd.DataFrame(
        columns=[
            "symbol",
            "secType",
            "exchange",
            "position",
            "marketPrice",
            "marketValue",
            "averageCost",
            "unrealizedPNL",
            "realizedPNL",
            "accountName",
        ]
    )

    # Instantiate the app, connect to TWS, and run the app
    app = PortfolioInfoApp(account_value_df, portfolio_df)
    app.connect("127.0.0.1", port, 0)
    app.run()

    # TODO: figure out why net liquidation is not simply the sum of total cash value and stock market value

    # Drop all values except those pertaining to account market value. Then convert these to float.
    account_value_df = (
        account_value_df.drop_duplicates()
        .loc[
            (
                account_value_df["key"].isin(
                    ["AvailableFunds", "NetLiquidation", "TotalCashValue", "StockMarketValue"]
                )
            )
            & (account_value_df["currency"] == "USD")
        ]
        .reset_index(drop=True)
    )
    account_value_df["value"] = account_value_df["value"].astype(float)

    # Calculate total account value and cash holdings
    account_value = np.round(
        account_value_df.loc[account_value_df["key"] == "NetLiquidation", "value"].values[0], 2
    )
    cash_value = np.round(
        account_value_df.loc[account_value_df["key"] == "TotalCashValue", "value"].values[0], 2
    )

    # Create portfolio holdings dataframe, convert 'position' to float, and reset the index
    port_holdings = portfolio_df.drop_duplicates()[
        ["symbol", "position", "marketPrice", "marketValue"]
    ]
    port_holdings["position"] = port_holdings["position"].astype(float)
    port_holdings.set_index("symbol", inplace=True)

    # Add in cash as a holding
    port_holdings.loc["$USD"] = [cash_value, 1, cash_value]

    # Calculate weight of each holding
    port_holdings["weight"] = port_holdings["marketValue"] / account_value

    # Round all values to 2 decimal places
    port_holdings = np.round(port_holdings, 2)

    # Return the resulting DataFrame
    return port_holdings
