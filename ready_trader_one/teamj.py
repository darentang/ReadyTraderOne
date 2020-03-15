import asyncio

from typing import List, Tuple
import itertools
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

class Orderbook:
    def __init__(self, name):
        self.bid_volumes = None
        self.ask_volumes = None
        self.bid_prices = None
        self.ask_prices = None
        self.name = name
        self.figure = plt.figure()
        self.max_length = 50
        self.history = np.zeros((self.max_length, 2))
        self.gradient_length = 50
        self.fit_degree = 2
        self.fit_coeff = np.zeros(self.fit_degree + 1)
        self.i = 0

    def get_table(self):
        return np.vstack((self.bid_volumes, self.bid_prices, self.ask_prices, self.ask_volumes)).astype(str).T

    def update_orders(self, bid_volumes, bid_prices, ask_prices, ask_volumes, time=0):
        self.bid_volumes = bid_volumes
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.ask_prices = ask_prices
        self.total_volume = np.sum([bid_volumes, ask_volumes])
        
        self.update(self.midpoint(), time)
        

    def update(self, price, time):
        if self.i < self.max_length:
            self.history[self.i, :] = [time, price]
        else:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1, :] = [time, price]
        self.fit()
        self.i += 1

    def gradient(self, time):
        return np.polyval(np.polyder(self.fit_coeff, 1), time)

    def acceleration(self, time):
        return np.polyval(np.polyder(self.fit_coeff, 2), time)

    def predict(self, time):
        return np.polyval(self.fit_coeff, time)

    def fit(self):
        if self.i < self.fit_degree:
            return

        if self.i > self.max_length:
            i = self.max_length - 1
        else:
            i = self.i
        
        if self.i > self.gradient_length:
            l = self.gradient_length - 1
        else:
            l = i
        
        
        X = self.history[i  - l:i + 1, 0]
        Y = self.history[i  - l:i + 1, 1]
        self.fit_coeff = np.polyfit(X, Y, deg=self.fit_degree)

    def best_bid(self):
        return self.bid_prices[0]

    def best_ask(self):
        return self.ask_prices[0]

    def midpoint(self):
        if self.total_volume != 0:
            return (np.dot(self.bid_prices, self.bid_volumes) + np.dot(self.ask_prices, self.ask_volumes)) / self.total_volume
        else:
            return 0

class Constants:
    ETA = - 0.005
    MAX_ORDER = 15
    MAX_VOLUME = 100
    TIMEOUT = 1.0
    INVENTORY_THRESHOLD = 50
    # Acceleration
    ALPHA = 1.0 
    # Speed
    BETA = 0.2
    # Discount for inventory
    KAPPA = 0.012
    SPEED = 20.0
    # maximum message per second
    MAX_MESSAGE = 20

class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = 0

        self.start_time = self.event_loop.time()

        self.ask_volume = self.bid_volume = 0
        self.ask_changed = self.bid_changed = False

        self.bid_time = 0
        self.ask_time = 0
        self.update_time = 0

        self.constants = Constants

        self.etf_position = self.future_position = 0

        self.high = self.low = self.mean = 0
        self.orderbook = Orderbook("ETF")

        self.speed = self.constants.SPEED

        self.command_buffer = []

        self.prices = {
        Instrument.ETF:{
                "mean": 0,
                "high": 0,
                "low": 0
            },
        Instrument.FUTURE:{
                "mean": 0,
                "high": 0,
                "low": 0
            },
            
        }
        

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.on_order_status_message(client_order_id, 0, 0, 0)


    def inventory(self, side):
        if self.etf_position + self.bid_volume > -self.etf_position + self.ask_volume:
            q = self.etf_position + self.bid_volume
        else:
            q = self.etf_position - self.ask_volume
        remaining_volume = np.min((self.constants.MAX_ORDER, self.constants.MAX_VOLUME - np.abs(q)))
        opposing_volume = self.constants.MAX_ORDER
        if q < 0:
            bid = opposing_volume
            ask = remaining_volume 
        else:
            bid = remaining_volume 
            ask = opposing_volume

        if side == Side.BUY:
            return int(bid)
        elif side == Side.SELL:
            return int(ask)
        

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
            self.bid_price = price
            self.bid_volume = volume
            self.bid_time = self.get_time()
            order_id = self.bid_id
        elif side ==  Side.SELL:
            self.ask_id = next(self.order_ids)
            self.ask_price = price
            self.ask_volume = volume
            self.ask_time = self.get_time()
            order_id = self.ask_id
        else:
            self.logger.info("Invalid side %d", side)
            return
        
        if len(self.command_buffer) + 1 < self.constants.MAX_MESSAGE:
            self.send_insert_order(order_id, side, price, volume, lifspan)
            self.command_buffer.append(self.get_time())
            self.logger.info("(id: %d) Placing  %s %d lots for $%d", order_id, "bid" if side == Side.BUY else "ask", volume, price // 100)


    def cancel(self, order_id):
        if order_id > 0:
            if len(self.command_buffer) + 1 < self.constants.MAX_MESSAGE:
                self.send_cancel_order(order_id)
                self.command_buffer.append(self.get_time())
                self.logger.info("Cancelling %d", order_id)
    
    def pricing(self, side):
        discount = 0 
        if self.etf_position < 0:
            discount =  np.exp(self.constants.KAPPA * np.abs(self.etf_position)) * 100 - 100
        elif self.etf_position > 0:
            discount = - np.exp(self.constants.KAPPA * np.abs(self.etf_position)) * 100 + 100

        # ask
        if side == Side.SELL:
            # price = self.high + self.constants.ALPHA * np.max((0, self.delta_m)) / 10 / self.speed + self.constants.BETA * np.max((0, self.orderbook.gradient())) * 10000 / self.speed
            price = np.max((self.prices[Instrument.ETF]["high"], self.prices[Instrument.FUTURE]["high"], self.best_ask))
            
            # if self.etf_position > 0:
            price += discount
            # price = np.max((price, self.prices[Instrument.ETF]["high"]))
            time = self.ask_time

        # bid
        elif side == Side.BUY:
            # price = self.low + self.constants.ALPHA * np.min((0, self.delta_m)) / 10 / self.speed + self.constants.BETA * np.min((0, self.orderbook.gradient())) * 10000 / self.speed
            price = np.min((self.prices[Instrument.ETF]["low"], self.prices[Instrument.FUTURE]["low"], self.best_bid))

            # if self.etf_position < 0:
            price += discount
            # price = np.min((price, self.prices[Instrument.ETF]["low"]))
            time = self.bid_time

        # delta = self.orderbook.gradient(self.get_time()) * (self.get_time() - time) * self.constants.BETA
        # self.logger.info("Current delta is %.2f", delta)
        # price -= delta
        
        return int(price // 100) * 100


    def time_out(self, side):
        if side == Side.SELL:
            return self.get_time() - self.ask_time > self.constants.TIMEOUT
        elif side == Side.BUY:
            return self.get_time() - self.bid_time > self.constants.TIMEOUT

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.
        
        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.

        TeamJ Strategy:

        Put in orders when there are changes to the highs and lows

        Update orders if something has been fileld?

        If gradient is negative, increase spread but maintain ask price

        If gradient is positive, increase spread but maintain bid price

        Look at volume detection later

        """
        

        if np.sum(ask_volumes) + np.sum(bid_volumes) == 0:
            return
        
        if instrument == Instrument.FUTURE:
            self.best_bid = bid_prices[0]
            self.best_ask = ask_prices[0] 


        if self.bid_id == 0 or self.bid_changed or self.ask_changed or self.time_out(Side.BUY):
            pricing = self.pricing(Side.BUY)
            self.logger.info("Quoting bid price %d (Existing %d)", pricing // 100, self.bid_price // 100)
            if pricing > self.ask_price:
                self.logger.info("Bid crosses ask, cancelling ask")
                self.cancel(self.ask_id)
            if pricing != self.bid_price:
                self.logger.info("placing bid")
                self.cancel(self.bid_id)
                self.insert(Side.BUY, self.inventory(Side.BUY), pricing)
            else:
                self.logger.info("Same price as existing bid")

        if self.ask_id == 0 or self.bid_changed or self.ask_changed or self.time_out(Side.SELL):
            pricing = self.pricing(Side.SELL)
            self.logger.info("Quoting ask price %d (Existing %d)", pricing // 100, self.ask_price // 100)
            if pricing < self.bid_price:
                self.logger.info("Ask crosses bid, cancelling bid")
                self.cancel(self.bid_id)
            if pricing != self.ask_price:
                self.logger.info("placing ask")
                self.cancel(self.ask_id)
                self.insert(Side.SELL, self.inventory(Side.SELL), pricing)
            else:
                self.logger.info("Same price as existing ask")

        self.ask_changed = self.bid_changed = False
        # print(self.orderbook.history)
        self.update_buffer()

    def update_buffer(self):
        self.command_buffer = [x for x in self.command_buffer if self.get_time() - x < 1.0]
        
    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        

        # only when it cancels
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.bid_volume = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.ask_volume = 0
            if fill_volume == 0:
                self.logger.info("Order %d cancelled", client_order_id)
            else:
                self.logger.info("Order %d filled (%d / %d)", client_order_id, fill_volume, fill_volume + remaining_volume)
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

        

    def on_trade_ticks_message(self, instrument: int, trade_ticks: List[Tuple[int, int]]) -> None:
        """Called periodically to report trading activity on the market.

        Each trade tick is a pair containing a price and the number of lots
        traded at that price since the last trade ticks message.
        """

        transactions = np.array(trade_ticks)
        mean = np.dot(transactions[:, 1], transactions[:, 0]) / np.sum(transactions[:, 1])            
        high = np.max(transactions[:, 0])
        low = np.min(transactions[:, 0])

        if high == low:
            high += 100

        self.prices[instrument]['mean'] = mean
        self.prices[instrument]['high'] = high
        self.prices[instrument]['low'] = low

        self.update_time = self.get_time()

        if instrument == Instrument.ETF:
            self.orderbook.update(mean, self.update_time)

