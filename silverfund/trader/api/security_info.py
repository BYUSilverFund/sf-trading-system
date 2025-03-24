from threading import Timer
from time import sleep

import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class SecurityInfo(EWrapper, EClient):
    def __init__(self, security_info_df, ticker_list):
        EClient.__init__(self, self)
        self.security_info_df = security_info_df
        self.ticker_list = ticker_list
        self.loaded_count = 0
        self.n = len(ticker_list)
        self.warned = False

    def nextValidId(self, orderId):

        for i, ticker in enumerate(self.ticker_list):

            # Build contract
            contract = Contract()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.primaryExchange = "ISLAND"
            contract.currency = "USD"
            contract.symbol = ticker

            self.reqContractDetails(i, contract)

        if len(self.security_info_df) == len(self.ticker_list):
            self.disconnect()
        else:
            self.loaded_count = len(self.security_info_df)
            Timer(3, self.wrap_up).start()

    def wrap_up(self):
        print(f"({len(self.security_info_df)}/{self.n})")
        if len(self.security_info_df) == len(self.ticker_list):
            self.disconnect()
        elif self.loaded_count == len(self.security_info_df):
            if self.loaded_count < 0.9 * len(self.ticker_list):
                if not self.warned:
                    print(
                        "Warning: Several seconds have elapsed since Trader Workstation has"
                        + " transmitted data. If the program appears stuck, terminate the program,"
                        + " restart Trader Workstation, and try once again."
                    )
                    self.warned = True
                self.loaded_count = len(self.security_info_df)
                Timer(3, self.wrap_up).start()
            else:
                self.disconnect()
        else:
            self.loaded_count = len(self.security_info_df)
            Timer(3, self.wrap_up).start()

    def contractDetails(self, reqId, contractDetails):
        self.security_info_df.loc[self.ticker_list[reqId]] = contractDetails.minTick

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Overrides default error handling."""

        if errorCode in [504, 2104, 2106, 2158, 10167]:

            # These codes simply indicate that the API made connection well. Do nothing.
            return

        return super().error(errorCode, errorString, advancedOrderRejectJson)


def get_security_info(ticker_list, port=7496):

    security_info_df = pd.DataFrame(columns=["min_tick"])

    print(
        f"Now retrieving minimum price increment allowed for each of the {len(ticker_list)} securities to be ordered."
    )

    app = SecurityInfo(security_info_df, ticker_list)
    app.connect("127.0.0.1", port, 0)
    app.run()

    print()

    no_min_tick_list = [ticker for ticker in ticker_list if ticker not in security_info_df.index]

    if no_min_tick_list:
        print(
            f"Failed to verify minimum limit price increment details for the following {len(no_min_tick_list)} securities\n"
        )
        print(no_min_tick_list)
        print()
        print("Using default minimum price increment of 0.01.")
        for ticker in no_min_tick_list:
            security_info_df.loc[ticker] = 0.01
    else:
        print("Minimum price increment retrieval successful.")

    print()

    return security_info_df
