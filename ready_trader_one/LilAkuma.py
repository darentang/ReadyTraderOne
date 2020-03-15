import asyncio

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

import itertools
import numpy as np

from config import speed

class Constants:
    GAMMA = 0.01
    KAPPA = 2
    ETA = - 0.005
    MAX_ORDER = 50
    DEFAULT_T = 0.25
    MAX_VOLUME = 95
    TIME_OUT = 1.0
    UPDATE = 1.0
    MAX_MESSAGE = 20


class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_price = self.bid_price = 0
        self.constants = Constants

        self.start_time = self.event_loop.time()


        # initialise positions
        self.etf_position = self.future_position = 0

        # speed of the simulation
        self.speed = speed

        # bid prices
        self.bid_price = self.ask_price = 0

        self.bids = []
        self.asks = []

        self.fill_times = [0, 0]


    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.on_order_status_message(client_order_id, 0, 0, 0)


    
    def get_time(self):
        # not accounting for speed of the engine tho....
        elapsed_time = self.event_loop.time() - self.start_time
        elapsed_time *= self.speed
        return elapsed_time



    def insert(self, side, volume, price, lifspan=Lifespan.GOOD_FOR_DAY):
        order_id = 0
        if volume <= 0 or price <= 0:
            self.logger.info("Invalid volume or price. V: %d P: %d", volume, price)
            return

        if side == Side.BUY:
            self.bid_id = next(self.order_ids)
            order_id = self.bid_id
            self.bids.append(order_id)
        elif side ==  Side.SELL:
            self.ask_id = next(self.order_ids)
            order_id = self.ask_id
            self.asks.append(order_id)
        else:
            self.logger.info("Invalid side %d", side)
            return

        self.send_insert_order(order_id, side, price, volume, lifspan)
        self.logger.info("(id: %d) Placing  %s %d lots for $%d", order_id, "bid" if side == Side.BUY else "ask", volume, price // 100)



    def cancel(self, ids):
        for i in ids:
            self.send_cancel_order(i)
            self.logger.info("Cancelling bid %d", i)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """

        # turn everything into numpy arrays
        bid_prices = np.array(bid_prices).astype(np.int64)
        ask_prices = np.array(ask_prices).astype(np.int64)
        bid_volumes = np.array(bid_volumes).astype(np.int64)
        bid_volumes = np.array(bid_volumes).astype(np.int64)

        # midpoint price 
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]

        self.total_volume = np.sum([bid_volumes, ask_volumes])
        if self.total_volume == 0 or best_bid == 0 or best_ask == 0:
            return
       

        if instrument == Instrument.FUTURE:
            # if no order in book
            if self.etf_position + 6 < self.constants.MAX_VOLUME and self.get_time() - self.fill_times[0] > self.constants.TIME_OUT:
                if len(self.bids) == 0:
                    self.insert(Side.BUY, 1, best_bid - 100)
                    self.insert(Side.BUY, 2, best_bid - 200)
                    self.insert(Side.BUY, 3, best_bid - 300)
                elif best_bid != self.bid_price:
                    self.bid_price = best_bid
                    self.cancel(self.bids)
                    self.insert(Side.BUY, 1, best_bid - 100)
                    self.insert(Side.BUY, 2, best_bid - 200)
                    self.insert(Side.BUY, 3, best_bid - 300)

            if self.etf_position - 6 > - self.constants.MAX_VOLUME and self.get_time() - self.fill_times[1] > self.constants.TIME_OUT:
                if len(self.asks) == 0:
                    self.insert(Side.SELL, 1, best_ask + 100)
                    self.insert(Side.SELL, 2, best_ask + 200)
                    self.insert(Side.SELL, 3, best_ask + 300)
                elif best_ask != self.ask_price:
                    self.ask_price = best_ask
                    self.cancel(self.asks)
                    self.insert(Side.SELL, 1, best_ask + 100)
                    self.insert(Side.SELL, 2, best_ask + 200)
                    self.insert(Side.SELL, 3, best_ask + 300)


                
    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """


        # on completing an order
        if remaining_volume == 0:
            if client_order_id in self.bids:
                self.bids.remove(client_order_id)
                self.fill_times[0] = self.get_time()
            if client_order_id in self.asks:
                self.asks.remove(client_order_id)
                self.fill_times[1] = self.get_time()

            self.logger.info("Order %d cancelled", client_order_id)
        elif fill_volume == 0:
            self.logger.info("Order %d inserted", client_order_id)
        else:
            self.logger.info("Order %d filled (%d / %d)", client_order_id, fill_volume, fill_volume + remaining_volume)

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


    