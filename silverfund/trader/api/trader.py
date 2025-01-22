from threading import Timer
from time import sleep

import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import *
from ibapi.wrapper import EWrapper


class Trader(EWrapper, EClient):
    """
    Places orders for each change in position requested by rebalance_port.
    Internal use only.
    """

    def __init__(self, order_df, error_list, quantity_col="quantity"):

        EClient.__init__(self, self)
        self.order_df = order_df.copy()
        self.error_list = error_list
        self.quantity_col = quantity_col
        self.bid_price_df = pd.DataFrame(columns=["bid"])
        self.ask_price_df = pd.DataFrame(columns=["ask"])
        self.firstOrderId = 0
        self.can_disconnect = True
        self.received_bid_idx_list = []
        self.received_ask_idx_list = []
        self.second_try = True

    def nextValidId(self, orderId):
        """Automatically executed whenever the app is run. Starts the order process"""

        # Set attributes to keep track of the number of orders as well as the first and current Ids
        self.nextOrderId = orderId
        self.firstOrderId = orderId
        self.order_count = 0

        # Start the order process
        self.start()

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):

        if errorCode in [399, 504, 2104, 2106, 2158, 2161, 10167]:
            return  # These codes do not affect the transmission of the order. Do nothing.

        else:

            # Other errors may or may not affect the transmission of the order.
            # Record the ticker for bookkeeping, then display the default error message for info.
            self.can_disconnect = False
            Timer(5, self.allow_disconnect).start()
            if reqId >= self.firstOrderId:
                ticker = self.order_df.index.values[reqId - self.firstOrderId]
                self.error_list.append(ticker)
                return super().error(
                    errorCode, ticker + ": " + errorString, advancedOrderRejectJson
                )
            else:
                return super().error(errorCode, errorString, advancedOrderRejectJson)

    def allow_disconnect(self):
        self.can_disconnect = True

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 1:  # 66 corresponds to the delayed bid price, 1 for live
            self.bid_price_df.loc[self.order_df.index.values[reqId]] = price
            self.received_bid_idx_list.append(reqId)
        if tickType == 2:  # 67 corresponds to the delayed ask price, 2 for live
            self.ask_price_df.loc[self.order_df.index.values[reqId]] = price
            self.received_ask_idx_list.append(reqId)
            # Result: self.price_df now contains a new entry with the ticker as the index

    def start(self):
        """Iterates through each requested order, placing the order in TWS."""

        self.order_df["ruleId"] = 0
        # Build each contract
        self.contracts = []
        order_ticker_list = self.order_df.index.values
        self.n = len(order_ticker_list)  # for progress updates
        for ticker in order_ticker_list:
            contract = Contract()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contract.primaryExchange = "NASDAQ"
            contract.symbol = ticker
            self.contracts.append(contract)

        print("Now calculating limit prices for each security:")

        self.reqMarketDataType(1)
        for i, contract in enumerate(self.contracts):

            self.reqMktData(i, contract, "", False, False, [])

            if i % 100 == 99:
                for j in range(i - 99, i + 1):
                    self.cancelMktData(j)

        self.prev_count = min(len(self.bid_price_df), len(self.ask_price_df))
        Timer(5, self.collect_prices).start()

    def collect_prices(self):
        price_count = min(len(self.bid_price_df), len(self.ask_price_df))
        print(f"({price_count}/{self.n})")
        if price_count == self.n:
            self.execute()
        elif price_count == self.prev_count:
            # TODO: Wouldn't this bypass some stocks on an edge case?
            if self.second_try:
                self.execute()
            else:
                self.try_again()
        else:
            self.prev_count = price_count
            Timer(5, self.collect_prices).start()

    # TODO: fix the iterating style of this class. At least if we are goin to let it have more than one retry
    def try_again(self):
        self.second_try = True

        second_try_idx_list = [
            idx
            for idx in range(len(self.contracts))
            if idx not in self.received_bid_idx_list or idx not in self.received_ask_idx_list
        ]
        print(second_try_idx_list)

        for i in second_try_idx_list:

            contract = self.contracts[i]
            print(contract.symbol)

            self.reqMktData(self.n + i, contract, "", False, False, [])

        sleep(1)

        for i in second_try_idx_list:
            self.cancelMktData(self.n + i)

        self.prev_count = min(len(self.bid_price_df), len(self.ask_price_df))
        Timer(5, self.collect_prices).start()

    def execute(self):

        print("Now executing orders for each security:")

        # Now place an order for each contract using the quantities from self.order_df
        for j, quantity in enumerate(self.order_df[self.quantity_col]):
            order = Order()
            ticker = self.contracts[j].symbol
            if ticker in self.bid_price_df.index and ticker in self.ask_price_df.index:
                lmtPrice = (
                    self.bid_price_df.loc[ticker, "bid"] + self.ask_price_df.loc[ticker, "ask"]
                ) / 2
            # If we can't fetch, we just recieve marketPrice. TODO: in conjuction with above, but fix if keep above.
            else:
                lmtPrice = self.order_df.loc[ticker, "marketPrice"]
            if lmtPrice <= 0:
                alt_price = self.order_df.loc[ticker, "marketPrice"]
                if alt_price > 0:
                    lmtPrice = alt_price
                else:
                    print(
                        f"ERROR: Calculated limit price for {ticker} was {lmtPrice}. Dropping from order"
                    )
                    self.error_list.append(ticker)
                    self.n -= 1
                    continue

            min_tick = self.order_df.loc[ticker, "min_tick"]
            if min_tick < 0.01 and lmtPrice > 1:
                min_tick = 0.01

            # Determine whether to buy or sell
            if quantity < 0:
                order.action = "SELL"

            else:
                order.action = "BUY"

            # Specify all other details of the order
            order.totalQuantity = int(abs(quantity))
            order.orderType = "LMT"
            order.lmtPrice = round(lmtPrice / min_tick) * min_tick
            order.eTradeOnly = False
            order.firmQuoteOnly = False

            # Place the order, then pause briefly to avoid sending too many requests at once
            self.placeOrder(self.nextOrderId, self.contracts[j], order)

            # Provide a progress report every 100 orders
            if self.order_count % 100 == 99:
                print(f"({self.order_count + 1}/{self.n})")

            # Increment the orderId and order_count for the next order
            self.nextOrderId += 1
            self.order_count += 1

        # Give a final progress update that all orders have been executed
        print(f"({self.n}/{self.n})\n")

        # Allow a little extra time for all orders to go through, then disconnect.
        sleep(10)
        while not self.can_disconnect:
            sleep(3)
        self.disconnect()

    def stop(self):
        self.done = True
        self.disconnect()
