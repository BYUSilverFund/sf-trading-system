from threading import Timer
from time import sleep

import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class LastPrice(EWrapper, EClient):
    """
    Uses TWS market data to retrieve and store the last price for each security.
    Internal use only.

    Parameters
    ----------
    price_df: DataFrame
        An empty dataframe to hold tickers and last prices
    ticker_list: list
        A list of all securities for which the app should find the last price
    invalid_ticker_list: list
        An empty list to hold tickers that cannot be identified by TWS
    """

    def __init__(
        self, last_price_df, ticker_list, invalid_ticker_list, account_type, second_try=False
    ):

        EClient.__init__(self, self)
        self.last_price_df = last_price_df
        self.ticker_list = ticker_list
        self.invalid_ticker_list = invalid_ticker_list
        if account_type == "paper":
            self.market_data_type = 3
            self.bid_tick_type = 68
            self.ask_tick_type = 67
        else:
            self.market_data_type = 1
            self.bid_tick_type = 4
            self.ask_tick_type = 2
        self.reqId = 0  # Initial index for market data requests
        self.n = len(ticker_list) + len(last_price_df)  # Total number of securities needing data
        # Note: On second pass, invalid tickers are not included in this total
        self.second_try = second_try

    def nextValidId(self, orderId: int):
        """Main function. Automatically executed when the app is run."""

        self.reqMarketDataType(
            self.market_data_type
        )  # Sets all data requests to request DELAYED data

        print(
            f"Retrieving last price data for {len(self.ticker_list)} securities. Approximate wait time: {round(0.04*len(self.ticker_list))} seconds"
        )
        # TODO: Update/verify approximation by testing on different systems with different combinations

        # Request market data for each security in the ticker list
        for ticker in self.ticker_list:

            # Build contract
            contract = Contract()
            contract.secType = "STK"
            contract.exchange = "NYSE"
            # contract.primaryExchange = 'ISLAND'
            contract.currency = "USD"
            contract.symbol = ticker

            # Request data
            self.reqMktData(self.reqId, contract, "", False, False, [])

            # TWS limits simultaneous market data subscriptions to just over 100.
            # Thus, cancel all active subscriptions after each set of 100 requests
            if self.reqId % 100 == 99:
                for i in range(self.reqId - 99, self.reqId + 1):
                    self.cancelMktData(i)
            # elif self.reqId == self.n - 1:
            #     sleep(1)
            #     for i in range((self.reqId // 100) * 100, self.n):
            #         self.cancelMktData(i)

            # Increment the request ID for the next security
            self.reqId += 1

        # Instantiate a new attribute that keeps track of how many ticks have been received so far
        # Each time a tick comes in, self.tickPrice adds the ticker and price to self.price_df
        # Thus, the length of self.price_df (originally empty) is the number of ticks registered
        self.previous_tick_count = len(self.last_price_df)

        # Allow five seconds for ticks to come in before calling self.wrap_up
        Timer(10, self.wrap_up).start()

    def tickPrice(self, reqId, tickType, price, attrib):
        """Saves the last traded price to self.last_price_df when a tick comes in

        This function is automatically called by TWS while market data for a security is
        requested. A market tick delivers information about a particular security and its
        prices. It is essential that sufficient time is given for each tick to be registered
        so that the following code can be executed for each security.
        """
        if self.second_try:
            if (
                tickType == self.ask_tick_type
            ):  # 67 corresponds to the delayed ask price, 2 for live
                if price <= 0:
                    print(f"ERROR: Retrieved price of {price} for {self.ticker_list[reqId]}.")
                else:
                    self.last_price_df.loc[self.ticker_list[reqId]] = price
        else:
            if (
                tickType == self.bid_tick_type
            ):  # 68 corresponds to the delayed last price, 4 for live
                if price <= 0:
                    print(f"ERROR: Retrieved price of {price} for {self.ticker_list[reqId]}.")
                else:
                    self.last_price_df.loc[self.ticker_list[reqId]] = price

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Overrides default error handling."""

        if errorCode in [504, 2104, 2106, 2158, 10167]:

            # These codes simply indicate that the API made connection well. Do nothing.
            return

        if errorCode == 200:

            """This error code indicates that TWS cannot find the contract requested. This is
            likely due to a buyout, name change, or merger. It may also be that the security's
            contract details are incorrect (for example, maybe the stock is trading in EUR
            instead of USD)."""

            # Add the corresponding tickers to the list of invalid tickers
            self.invalid_ticker_list.append(self.ticker_list[reqId])
            return

        if errorCode == 300:

            # This error occurs when trying to cancel a subscription that was never
            # activated because the ticker was invalid. Ignore it in this case.
            if self.ticker_list[reqId] in self.invalid_ticker_list:
                return

        if errorCode == 321:
            print(self.ticker_list[reqId])

        # If not any of the previous error codes, revert to default error handling
        return super().error(errorCode, errorString, advancedOrderRejectJson)

    def wrap_up(self):
        """Closes the app once no more new price ticks appear in a 5-second period"""

        current_count = len(self.last_price_df)
        if current_count == self.previous_tick_count:

            # If the current number of ticks, (len(self.price_df)), matches the previously
            # stored value, no new ticks have come in during the last 5 seconds. End the program.
            self.disconnect()

        else:  # At least one new tick has come in.

            # Store the current number of ticks
            self.previous_tick_count = current_count

            # Report the number of ticks received so far out of the total
            print(f"({current_count}/{self.n})")

            # Call the function again after another five seconds
            Timer(5, self.wrap_up).start()


def get_last_price(ticker_list, account_type, port=7496):
    """Calls LastPrice TWS API app to get closing prices for each security in ticker_list."""

    # Initialize an empty dataframe to hold closing prices and a list to hold unrecognized tickers
    last_price_df = pd.DataFrame(columns=["price"])
    invalid_ticker_list = []

    # Create an instance of the LastPrice class, connect to the TWS, and run the app
    app = LastPrice(last_price_df, ticker_list, invalid_ticker_list, account_type)
    app.connect("127.0.0.1", port, 0)
    app.run()

    if len(last_price_df) == 0:
        print(
            "\nCould not access market data!\nPlease check that markets are open "
            + "and that you are properly connected to Trader Workstation. Then run the program again."
        )
        quit()

    # Inform the user of any tickers that were not recogized by TWS
    if invalid_ticker_list != []:

        print(
            f"\nThe following {len(invalid_ticker_list)} tickers are not currently valid in Trader Workstation:\n\n",
            invalid_ticker_list,
            "\n",
        )

    # Calculate the number of securities to be dropped due to missing data
    no_data_count = len(ticker_list) - len(invalid_ticker_list) - len(last_price_df)
    done = False

    while no_data_count != 0 and not done:

        go_on = input(
            f"\nFailed to get last price data for {no_data_count} securities.\n"
            + "Try again for these securities? Y/N "
        )
        if go_on != "Y" and go_on != "y":
            done = True

        else:
            # Create a list of all valid tickers for which data was not received successfully
            second_try_ticker_list = [
                ticker
                for ticker in ticker_list
                if ticker not in last_price_df.index and ticker not in invalid_ticker_list
            ]

            # Try to receive data for these securities a second time
            app = LastPrice(
                last_price_df, second_try_ticker_list, [], account_type, second_try=True
            )
            app.connect("127.0.0.1", port, 0)
            app.run()

            no_data_count = len(ticker_list) - len(invalid_ticker_list) - len(last_price_df)

    # Return the ticker list, invalid ticker list, and price DataFrame
    return last_price_df, ticker_list
