import asyncio

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

import itertools
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class Orderbook:
    def __init__(self, name, show_animation=False):
        self.bid_volumes = None
        self.ask_volumes = None
        self.bid_prices = None
        self.ask_prices = None
        self.name = name
        self.figure = plt.figure()
        self.show_animation = show_animation
        self.history = []
        self.max_length = 100
        self.gradient_length = 10

    def get_table(self):
        return np.vstack((self.bid_volumes, self.bid_prices, self.ask_prices, self.ask_volumes)).astype(str).T

    def update(self, bid_volumes, bid_prices, ask_prices, ask_volumes, time=0):
        self.bid_volumes = bid_volumes
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.ask_prices = ask_prices
        self.total_volume = np.sum([bid_volumes, ask_volumes])
        if self.show_animation:
            self.animate()
        
        self.history.append((time, self.midpoint()))
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def gradient(self):
        if len(self.history) > 2:
            P = np.array([x[1] for x in self.history])
            X = np.vstack((list(x[0] for x in self.history), np.ones(len(self.history))))
            A = P @ np.linalg.pinv(X)
            return A[0]
        return 0
    def best_bid(self):
        return self.bid_prices[0]

    def best_ask(self):
        return self.ask_prices[0]

    def midpoint(self):
        if self.total_volume != 0:
            return (np.dot(self.bid_prices, self.bid_volumes) + np.dot(self.ask_prices, self.ask_volumes)) / self.total_volume
        else:
            return 0
    def animate(self):
        plt.figure(1)
        plt.clf()

        # plt.table(cellText=self.get_table())
        plt.subplot(1, 2, 1)
        plt.barh(self.bid_prices/100, self.bid_volumes, align='center', color='b')
        plt.plot([0, 100],[self.midpoint() / 100, self.midpoint() / 100], c='g')
        plt.ylim(3500, 3600)
        plt.xlim(0, 120)
        plt.gca().invert_xaxis()
        plt.subplot(1, 2, 2)
        plt.barh(self.ask_prices/100, self.ask_volumes, align='center', color='r')
        plt.plot([0, 100],[self.midpoint() / 100, self.midpoint() / 100], c='g')
        plt.ylim(3500, 3600)
        plt.xlim(0, 120)

        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())


class Constants:
    GAMMA = 0.01
    KAPPA = 2
    ETA = - 0.005
    MAX_ORDER = 50
    DEFAULT_T = 0.25
    MAX_VOLUME = 100
    TIME_OUT = 2.0
    UPDATE = 2.0
    UNSTUCK_TIME = 2.0

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

        # TODO: CHANGE THIS
        # speed of the simulation
        self.speed = 1.0

        # store the historical midpoint data of ETF and FUTURE
        self.history = {
            Instrument.ETF: [],
            Instrument.FUTURE: []
        }

        # bid prices
        self.bid_price = self.ask_price = 0
        
        # time at which last position changed
        self.last_change = self.start_time

        self.orderbook = Orderbook("ETF", False)


    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.on_order_status_message(client_order_id, 0, 0, 0)

    def get_sigma(self, history):
        """
        Get volatility of the stock

        """
        if len(history) > 2:
            price_hist = np.array([x[1] for x in history])[1:]
            price_change = price_hist[:-1] - price_hist[1:]
            return np.std(price_change)
        
        return 0


    def inventory(self):
        q = self.etf_position
        remaining_volume = np.min((self.constants.MAX_ORDER, self.constants.MAX_VOLUME - np.abs(q)))
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

        placed = False

        if bid_volume > 0 and bid_price > 0:
            placed = True
            self.bid_id = next(self.order_ids)
            self.send_insert_order(self.bid_id, Side.BUY, bid_price, bid_volume, lifespan)
        else:
            self.bid_id = -1

        if ask_volume > 0 and ask_price > 0:
            placed = True
            self.ask_id = next(self.order_ids)
            self.send_insert_order(self.ask_id, Side.SELL, ask_price, ask_volume, lifespan)
        else:
            self.ask_id = -1
        
        # if nothing has been done
        if not placed:
            return

        self.quote_time = self.get_time()
        self.logger.info("Placing quotes (%d, %d). Bid: %d @ $%d, Ask: %d @ $%d", self.bid_id, self.ask_id, bid_volume, bid_price / 100, ask_volume, ask_price / 100)

    def cancel(self, order_id, side=""):
        self.logger.info("Cancelling %s %d", side, order_id)
        if side == "bid":
            self.bid_id = 0
        elif side == "ask":
            self.ask_id = 0
        if order_id > 1:
            self.send_cancel_order(order_id)
    
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


        if np.sum((bid_volumes, ask_volumes)) == 0 or best_bid == 0 or best_ask == 0:
            return

        self.orderbook.update(bid_volumes, bid_prices, ask_prices, ask_volumes, elapsed_time)
        # self.orderbook.gradient()
        if instrument == Instrument.FUTURE:
            # if no order in book
            if self.bid_id == 0 and self.ask_id == 0:
                # set to best bid and ask
                self.bid_price, self.ask_price = best_bid, best_ask

                # get volume
                self.bid_volume, self.ask_volume = self.inventory()

                # time at which we quote
                self.quote_time = self.get_time()

                # send quote
                self.quote()
            # if one order in book
            elif ( (self.bid_id != 0) ^ (self.ask_id != 0) ) and self.get_time() - self.execution_time > self.constants.TIME_OUT:
                # buy is not fufilled
                if self.bid_id != 0:
                    self.cancel(self.bid_id, "bid")
                    
                # sell is not fufilled
                else:
                    self.cancel(self.ask_id, "ask")
                
                # if we experience lot of fluctuation, dont do shit
                if np.abs(self.orderbook.gradient()) > 10:
                    return 

                # otherwise send a new quote with the half spread
                self.bid_price, self.ask_price = best_bid, best_ask
                self.bid_volume, self.ask_volume = self.inventory()
                half_spread = (best_ask - best_bid) * 0.5
                if self.bid_id != 0:
                    self.bid_price = best_ask - half_spread
                    self.ask_price += half_spread
                else:
                    self.bid_price -= half_spread
                    self.ask_price = best_bid + half_spread

                self.quote()

            # if two orders in book and timeout
            elif (self.bid_id != 0 and self.ask_id != 0) and elapsed_time - self.quote_time > self.constants.UPDATE:
                self.cancel(self.bid_id, "bid")
                self.cancel(self.ask_id, "ask")
    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """

        elapsed_time = self.get_time()

        # only when it cancels
        if remaining_volume == 0:
            self.logger.info("Order %d cancelled", client_order_id)
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.execution_time = elapsed_time
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.execution_time = elapsed_time
        elif fill_volume == 0:
            self.logger.info("Order %d placed", client_order_id)
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

        self.last_change = self.get_time()
        

    def on_trade_ticks_message(self, instrument: int, trade_ticks: List[Tuple[int, int]]) -> None:
        """Called periodically to report trading activity on the market.

        Each trade tick is a pair containing a price and the number of lots
        traded at that price since the last trade ticks message.
        """
        pass