import asyncio

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

import itertools
import numpy as np

# TODO: IMPORTANT!! get rid of this and paste the actual class
from alec2 import Orderbook

class Constants:
    GAMMA = 0.01
    KAPPA = 2
    ETA = - 0.05
    MAX_ORDER = 99
    DEFAULT_T = 0.25
    MAX_VOLUME = 95
    TIME_OUT = 1.0
    UPDATE = 1.0


class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_price = self.bid_price = 0
        self.constants = Constants

        # Set sufficiently large T
        self.T = self.constants.DEFAULT_T
        self.start_time = self.event_loop.time()

        # initialise times
        self.quote_time = self.start_time

        # initialise positions
        self.etf_position = self.future_position = 0

        # speed of the simulation
        self.speed = 1.0

        # the active quotes right now, contains order ids
        self.active_quotes = {
            "bids": {},
            "asks": {}
        }

        self.execution_time = self.start_time

        # bid prices
        self.bid_price = self.ask_price = 0

        self.orderbook = Orderbook("ETF", False)

        self.bid_volume = self.ask_volume = self.bid_price = self.ask_price = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.on_order_status_message(client_order_id, 0, 0, 0)

    def inventory(self):
        active_bid_volume = np.sum([x for i, x in self.active_quotes["bids"].items()])
        active_ask_volume = np.sum([x for i, x in self.active_quotes["asks"].items()])
        if self.etf_position > 0:
            q = self.etf_position + active_bid_volume
        else:
            q = self.etf_position - active_ask_volume

        if np.abs(q) > self.constants.MAX_VOLUME:
            remaining_volume = 0
        else:
            remaining_volume = np.min((self.constants.MAX_ORDER, self.constants.MAX_VOLUME - q))
        opposing_volume = self.constants.MAX_ORDER
        if q < 0:
            bid = opposing_volume
            ask = remaining_volume * np.exp(- self.constants.ETA * q)
        else:
            bid = remaining_volume * np.exp(self.constants.ETA * q)
            ask = opposing_volume
        return int(bid), int(ask)
    
    def get_time(self):
        # not accounting for speed of the engine tho....
        elapsed_time = self.event_loop.time() - self.start_time
        elapsed_time *= self.speed
        return elapsed_time

    def quote(self, bid_price=None, ask_price=None, bid_volume=None, ask_volume=None, lifespan=Lifespan.GOOD_FOR_DAY):
        """
        Place a quote

        """

        if bid_price is None:
            bid_price = self.bid_price
        
        if ask_price is None:
            ask_price = self.ask_price

        if bid_volume is None:
            bid_volume = self.bid_volume

        if ask_volume is None:
            ask_volume = self.ask_volume

        # multiple of tick size (100 cents)
        bid_price -= bid_price % 100
        ask_price -= ask_price % 100 
        bid_price = int(bid_price)
        ask_price = int(ask_price)

        bid_volume = int(bid_volume)
        ask_volume = int(ask_volume)

        placed = False

        if bid_volume > 0 and bid_price > 0 and len(self.active_quotes["bids"]) < 2:
            placed = True
            self.bid_id = next(self.order_ids)
            self.send_insert_order(self.bid_id, Side.BUY, bid_price, bid_volume, lifespan)
            self.active_quotes["bids"][self.bid_id] = bid_volume

        if ask_volume > 0 and ask_price > 0 and len(self.active_quotes["asks"]) < 2:
            placed = True
            self.ask_id = next(self.order_ids)
            self.send_insert_order(self.ask_id, Side.SELL, ask_price, ask_volume, lifespan)
            self.active_quotes["asks"][self.ask_id] = ask_volume
        
        # if nothing has been done
        if not placed:
            return

        # time at which we quote
        self.quote_time = self.get_time()
        self.logger.info("Placing quotes (%d, %d). Bid: %d @ $%d, Ask: %d @ $%d", self.bid_id, self.ask_id, bid_volume, bid_price / 100, ask_volume, ask_price / 100)

    def cancel(self, ids):
        for i in ids.keys():
            self.send_cancel_order(i)
            self.logger.info("Cancelling bid %d", i)

    def quote_double(self, volume_ratio=0.8, levels=1):
        self.quote(bid_price=self.bid_price, 
                   ask_price=self.ask_price, 
                   bid_volume=volume_ratio * self.bid_volume, 
                   ask_volume=volume_ratio * self.ask_volume)
        self.quote(bid_price=self.bid_price - levels*100, 
                   ask_price=self.ask_price + levels*100, 
                   bid_volume=(1 - volume_ratio) * self.bid_volume, 
                   ask_volume=(1 - volume_ratio) * self.ask_volume)

        self.logger.info("Active bids: %d, asks %d", len(self.active_quotes["bids"]), len(self.active_quotes["asks"]))        

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        
        # turn everything into numpy arrays
        bid_prices = np.array(bid_prices)
        ask_prices = np.array(ask_prices)
        bid_volumes = np.array(bid_volumes)
        bid_volumes = np.array(bid_volumes)

        elapsed_time = self.get_time()

        if self.T < elapsed_time:
            self.T = elapsed_time + self.constants.DEFAULT_T
        # midpoint price 
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        # midpoint price 
        if np.sum((bid_volumes, ask_volumes)) == 0 or best_bid == 0 or best_ask == 0:
            return

        # if instrument == Instrument.ETF:
        #     self.orderbook.update(bid_volumes, bid_prices, ask_prices, ask_volumes, elapsed_time)


        bid_queue = self.active_quotes["bids"]
        ask_queue = self.active_quotes["asks"]

        if instrument == Instrument.FUTURE:
            # if no order in book
            if len(bid_queue) == 0 and len(ask_queue) == 0:
                # set to best bid and ask
                self.bid_price, self.ask_price = best_bid, best_ask

                # get volume
                self.bid_volume, self.ask_volume = self.inventory()

                # send double-decker quotes
                self.quote_double()
            
            elif len(ask_queue) < 2 or len(bid_queue) < 2:
                self.logger.info("Changing quote. Bid queue: %d, Ask queue: %d", len(bid_queue), len(ask_queue))
                self.cancel(bid_queue)
                self.cancel(ask_queue)

                if np.abs(self.orderbook.gradient()) > 10:
                    return

                # otherwise send a new quote with the half spread
                self.bid_price, self.ask_price = best_bid, best_ask
                self.bid_volume, self.ask_volume = self.inventory()
                half_spread = (best_ask - best_bid) * 0.5
                if len(ask_queue) <  len(bid_queue):
                    self.bid_price = best_ask - half_spread
                    self.ask_price += half_spread
                elif len(ask_queue) >  len(bid_queue):
                    self.bid_price -= half_spread
                    self.ask_price = best_bid + half_spread
                else:
                    return

                self.quote_double()

            # if two orders in book and timeout
            elif (len(ask_queue) == 2 or len(bid_queue) == 2) and elapsed_time - self.quote_time > self.constants.UPDATE:
                self.logger.info("Timeout")
                self.cancel(bid_queue)
                self.cancel(ask_queue)
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
            if client_order_id in self.active_quotes['bids']:
                self.active_quotes['bids'].pop(client_order_id)
            
            if client_order_id in self.active_quotes['asks']:
                self.active_quotes['asks'].pop(client_order_id)

            self.logger.info("Order %d cancelled", client_order_id)
            self.execution_time = self.get_time()    
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
        # print(trade_ticks)
        pass
