import asyncio

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

import itertools
import numpy as np

class Constants:
    GAMMA = 0.01
    KAPPA = 2
    ETA = - 0.05
    MAX_ORDER = 50
    DEFAULT_T = 20
    MAX_VOLUME = 100
    TIME_OUT = 1.0
    UPDATE = 0.2

class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = 0
        self.constants = Constants

        # Set sufficiently large T
        self.T = self.constants.DEFAULT_T
        self.start_time = self.event_loop.time()

        # initialise times
        self.execution_time = self.start_time
        self.quote_time = self.start_time

        # initialise positions
        self.etf_position = self.future_position = 0

        # speed of the simulation
        self.speed = 5.0

        # store the historical midpoint data of ETF and FUTURE
        self.history = {
            Instrument.ETF: [],
            Instrument.FUTURE: []
        }

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.on_order_status_message(client_order_id, 0, 0, 0)

   

    def inventory(self):
        """
        Bid and ask volume based on existing position

        """
        q = self.etf_position
        remaining_volume = min((self.constants.MAX_ORDER, self.constants.MAX_VOLUME - abs(q)))
        opposing_volume = self.constants.MAX_ORDER
        if q < 0:
            bid = opposing_volume
            ask = remaining_volume * np.exp(- self.constants.ETA * q)
        else:
            bid = remaining_volume * np.exp(self.constants.ETA * q)
            ask = opposing_volume
        return int(bid), int(ask)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """

        # not accounting for speed of the engine tho....
        elapsed_time = self.event_loop.time() - self.start_time
        elapsed_time *= self.speed

        if self.T < elapsed_time:
            self.T = elapsed_time + self.constants.DEFAULT_T

        # midpoint price 
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        midpoint = 0.5 * (best_bid + best_ask)
        self.history[instrument].append((elapsed_time, midpoint))

        if midpoint == 0:
            return

        if instrument == Instrument.FUTURE:
            if self.bid_id == 0 and self.ask_id == 0:
                bid_price, ask_price = best_bid, best_ask
                bid_volume, ask_volume = self.inventory()
                self.bid_id = next(self.order_ids)
                self.ask_id = next(self.order_ids)
                self.send_insert_order(self.bid_id, Side.BUY, bid_price, bid_volume, Lifespan.GOOD_FOR_DAY)
                self.send_insert_order(self.ask_id, Side.SELL, ask_price, ask_volume, Lifespan.GOOD_FOR_DAY)
                self.quote_time = elapsed_time
                self.logger.info("Placing quotes (%d, %d). Bid: %d @ $%d, Ask: %d @ $%d", self.bid_id, self.ask_id, bid_volume, bid_price / 100, ask_volume, ask_price / 100)
            elif ( (self.bid_id != 0) ^ (self.ask_id != 0) ) and elapsed_time - self.execution_time > self.constants.TIME_OUT:
                # buy is not fufilled
                if self.bid_id != 0:
                    self.send_cancel_order(self.bid_id)
                    self.logger.info("Timeout, cancelling bid %d", self.bid_id)
                else:
                    self.send_cancel_order(self.ask_id)
                    self.logger.info("Timeout, cancelling ask %d", self.ask_id)
            elif (self.bid_id != 0 and self.ask_id != 0) and elapsed_time - self.quote_time > self.constants.UPDATE:
                self.send_cancel_order(self.bid_id)
                self.send_cancel_order(self.ask_id)
                self.logger.info("Timeout, cancelling bid %d and ask %d", self.bid_id, self.ask_id)
                



        # print(f"{elapsed_time:.2f} sec has elapsed")

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """

        # not accounting for speed of the engine tho....
        elapsed_time = self.event_loop.time() - self.start_time
        elapsed_time *= self.speed

        self.logger.info("Order %d filled (%d / %d)", client_order_id, fill_volume, fill_volume + remaining_volume)


        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.execution_time = elapsed_time
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.execution_time = elapsed_time
    


    def on_position_change_message(self, future_position: int, etf_position: int) -> None:
        """Called when your position changes.

        Since every trade in the ETF is automatically hedged in the future,
        future_position and etf_position will always be the inverse of each
        other (i.e. future_position == -1 * etf_position).
        """
        self.logger.info("Position changed to ETF: %d, FUTURE: %d", etf_position, future_position)
        self.etf_position = etf_position
        self.future_position = future_position
        

    def on_trade_ticks_message(self, instrument: int, trade_ticks: List[Tuple[int, int]]) -> None:
        """Called periodically to report trading activity on the market.

        Each trade tick is a pair containing a price and the number of lots
        traded at that price since the last trade ticks message.
        """
        pass
