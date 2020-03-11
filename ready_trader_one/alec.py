import asyncio

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

import itertools
import numpy as np

class Constants:
    GAMMA = 0.01
    KAPPA = 2
    ETA = - 0.005
    MAX_ORDER = 50
    DEFAULT_T = 0.25
    MAX_VOLUME = 100
    TIME_OUT = 1.0
    UPDATE = 1.0

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

        # whether existing quotes have been places
        self.quoted = False

        # bid prices
        self.bid_price = self.ask_price = 0

        self.wait = False

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


    def quote(self, bid_price=None, ask_price=None, bid_volume=None, ask_volume=None):
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
            self.send_insert_order(self.bid_id, Side.BUY, bid_price, bid_volume, Lifespan.GOOD_FOR_DAY)

        if ask_volume > 0 and ask_price > 0:
            placed = True
            self.ask_id = next(self.order_ids)
            self.send_insert_order(self.ask_id, Side.SELL, ask_price, ask_volume, Lifespan.GOOD_FOR_DAY)
        
        # if nothing has been done
        if not placed:
            return

        self.quote_time = self.get_time()
        self.logger.info("Placing quotes (%d, %d). Bid: %d @ $%d, Ask: %d @ $%d", self.bid_id, self.ask_id, bid_volume, bid_price / 100, ask_volume, ask_price / 100)

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

        self.total_volume = np.sum([bid_volumes, ask_volumes])
        if self.total_volume == 0 or best_bid == 0 or best_ask == 0:
            return

        # midpoint = 0.5 * (best_bid + best_ask)
        midpoint = (np.dot(bid_prices, bid_volumes) + np.dot(ask_prices, ask_volumes)) / self.total_volume
        
        self.history[instrument].append((elapsed_time, midpoint))

        if instrument == Instrument.FUTURE:
            # if no order in book
            if self.bid_id == 0 and self.ask_id == 0:
                print("Quoting 2")
                # set to best bid and ask
                self.bid_price, self.ask_price = best_bid, best_ask

                # get volume
                self.bid_volume, self.ask_volume = self.inventory()

                # send quote
                self.quote()
            # if one order in book
            elif ( (self.bid_id != 0) ^ (self.ask_id != 0) ):
                print("Quoting 1")
                self.bid_price, self.ask_price = best_bid, best_ask

                # buy is not fufilled
                if self.bid_id != 0:
                    self.send_cancel_order(self.bid_id)
                    self.logger.info("Cancelling bid %d", self.bid_id)
                    
                    half_spread = (best_ask - best_bid) * 0.5
                    self.bid_price -= half_spread
                    self.ask_price = self.bid_price + half_spread
                    self.bid_id = 0

                # sell is not fufilled
                else:
                    self.send_cancel_order(self.ask_id)
                    self.logger.info("Cancelling ask %d", self.ask_id)

                    half_spread = (best_ask - best_bid) * 0.5
                    self.bid_price = self.ask_price - half_spread
                    self.ask_price += half_spread
                    self.ask_id = 0
                self.execution_time = elapsed_time



                self.bid_volume, self.ask_volume = self.inventory()
                self.quote()

            # if two orders in book and timeout
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

        elapsed_time = self.get_time()

        if remaining_volume == 0:
            self.logger.info("Order %d cancelled", client_order_id)
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0
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

class KalmanFilter:
    def __init__(self, R, Q, X0):
        self.R = R
        self.Q = Q
        self.X = X0
        self.P = self.R.copy()
        self.H = np.array([1, 0])
        self.K = np.zeros(2)
        self.t = 0

    def gamma(self, t):
        return np.array([0.5 * (t - self.t) ** 2, (t - self.t)])
        
    def A(self, t):
        return np.array([[1, t - self.t], [0, 1]])
    
    def predict(self, t):
        # state vector 
        self.X = self.A(t) @ self.X
        self.P = self.A(t) @ self.P @ self.A(t).T + self.gamma(t) @ self.Q @ self.gamma(t).T

    def update(self, z, t):
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.X = self.X + self.K @ (z - self.H @ self.X)
        self.P = (np.identity(2) - self.K @ self.H) @ self.P
        self.t = t

    def cov(self, sigma1, sigma2):
        cov_matrix = np.outer([sigma1, sigma2], [sigma1, sigma2])
        return np.diag(np.diag(cov_matrix))




    