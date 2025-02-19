from threading import Timer
from time import sleep

import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.contract import *
from ibapi.order import *
from ibapi.wrapper import *


class MarginImpact(EClient, EWrapper):
    def __init__(self, order_df):
        EClient.__init__(self, self)
        self.order_df = order_df
        self.n = len(order_df)
        self.init_margin_change = 0
        self.maint_margin_change = 0
        self.equity_with_loan_change = 0
        self.current_init_margin = 0
        self.current_maint_margin = 0
        self.current_equity_with_loan = 0
        self.received_data_count = 0
        self.warned = False

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        os = orderState
        self.current_init_margin = os.initMarginBefore
        self.current_maint_margin = os.maintMarginBefore
        self.current_equity_with_loan = os.equityWithLoanBefore
        self.init_margin_change += float(os.initMarginChange)
        self.maint_margin_change += float(os.maintMarginChange)
        self.equity_with_loan_change += float(os.equityWithLoanChange)
        self.received_data_count += 1

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Overrides default error handling."""

        if errorCode in [504, 2104, 2106, 2158, 10167]:

            # These codes simply indicate that the API made connection well. Do nothing.
            return

        return super().error(errorCode, errorString, advancedOrderRejectJson)

    def nextValidId(self, orderId):
        self.orderId = orderId

        for i in range(self.n):

            ticker = self.order_df.index.values[i]
            quantity = self.order_df.loc[ticker, "quantity"]
            price = self.order_df.loc[ticker, "marketPrice"]
            min_tick = self.order_df.loc[ticker, "min_tick"]
            if min_tick < 0.01 and price > 1:
                min_tick = 0.01

            contract = Contract()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contract.primaryExchange = "NASDAQ"
            contract.symbol = ticker

            order = Order()
            order.orderType = "LMT"
            order.action = "SELL" if quantity < 0 else "BUY"
            order.totalQuantity = int(abs(quantity))
            if order.totalQuantity == 0:
                self.order_df.drop(ticker)
                self.n -= 1
                continue
            order.lmtPrice = round(price / min_tick) * min_tick
            order.eTradeOnly = False
            order.firmQuoteOnly = False
            order.whatIf = True

            self.placeOrder(self.orderId, contract, order)

            self.orderId += 1

        self.delayed_count = self.received_data_count
        Timer(5, self.wrap_up).start()

    def wrap_up(self):
        print(f"({self.received_data_count}/{self.n})")
        if self.received_data_count == self.n:
            self.end()
        elif self.delayed_count == self.received_data_count:
            if self.delayed_count < 0.9 * self.n:
                if not self.warned:
                    print(
                        "Warning: Several seconds have elapsed since Trader Workstation has"
                        + " accepted data. If the program appears stuck, terminate the program,"
                        + " restart Trader Workstation, and try once again."
                    )
                    self.warned = True
                self.delayed_count = self.received_data_count
                Timer(5, self.wrap_up).start()
            else:
                print(
                    f"Failed to verify margin effects for {self.n - self.received_data_count} orders.\n"
                )
                self.end()
        else:
            self.delayed_count = self.received_data_count
            Timer(5, self.wrap_up).start()

    def end(self):
        print()
        print(f"Current Initial Margin: {self.current_init_margin}")
        print(f"Current Maintenance Margin: {self.current_maint_margin}")
        print(f"Current Equity with Loan: {self.current_equity_with_loan}")
        print()

        print(f"Initial Margin Change: {self.init_margin_change}")
        print(f"Maintenance Margin Change: {self.maint_margin_change}")
        print(f"Equity with Loan Change: {self.equity_with_loan_change}")

        print()

        est_available_funds = (
            float(self.current_equity_with_loan)
            - float(self.current_init_margin)
            - self.init_margin_change
        )
        print(f"Estimated Available Funds after Trade: {est_available_funds}\n")

        self.disconnect()


def calc_margin_impact(order_df):

    print("Now calculating margin impact for the rebalanced portfolio:\n")

    app = MarginImpact(order_df)
    app.connect("127.0.0.1", 7496, 1000)
    app.run()

    goOn = input("Proceed? Y/N ")
    if goOn != "Y" and goOn != "y":
        quit()
    else:
        print()
        return order_df
