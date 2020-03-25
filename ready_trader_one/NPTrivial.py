import asyncio

from typing import List, Tuple
import itertools
import numpy as np


from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side

from config import speed

class Constants:
    MAX_ORDER = 30
    MAX_VOLUME = 90
    TIMEOUT = 0.5

    # Spread
    KAPPA = 1

    # TODO: Change in final version
    SPEED = speed
    # maximum message per second
    MAX_MESSAGE = 15
    GRADIENT_LENGTH = 5

class Orderbook:
    def __init__(self, name):
        self.bid_volumes = None
        self.ask_volumes = None
        self.bid_prices = None
        self.ask_prices = None
        self.constants = Constants
        self.name = name
        self.max_length = 50
        self.history = np.zeros((self.max_length, 2))
        self.gradient_length = self.constants.GRADIENT_LENGTH
        self.fit_degree = 1
        self.fit_coeff = np.zeros(self.fit_degree + 1)
        self.total_volume = 0
        self.bid_dict = {}
        self.ask_dict = {}
        self.i = 0

    def get_table(self):
        return np.vstack((self.bid_volumes, self.bid_prices, self.ask_prices, self.ask_volumes)).astype(str).T

    def update_orders(self, bid_volumes, bid_prices, ask_volumes, ask_prices, time=0):
        self.bid_volumes = bid_volumes
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.ask_prices = ask_prices

        self.bid_dict = dict(zip(bid_prices, bid_volumes))
        self.ask_dict = dict(zip(ask_prices, ask_volumes))

        self.total_volume = np.sum([bid_volumes, ask_volumes])

        self.update(self.midpoint(), time)

    def spread(self):
        return (self.best_ask() - self.best_bid())

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
        if self.bid_prices is not None:
            return self.bid_prices[0]
        else:
            return 0

    def best_ask(self):
        if self.ask_prices is not None:
            return self.ask_prices[0]
        else:
            return 0

    def midpoint(self):
        if self.total_volume != 0:
            return 0.5 * (self.best_ask() + self.best_bid())
            # return (np.dot(self.bid_prices, self.bid_volumes) + np.dot(self.ask_prices, self.ask_volumes)) / self.total_volume
        else:
            return 0

    def halfspread(self):
        return (self.best_ask() - self.best_bid()) / 2


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
        self.etf_orderbook = Orderbook("ETF")
        self.future_orderbook = Orderbook("FUTURE")

        self.speed = self.constants.SPEED

        self.command_buffer = []

        self.etf_prediction = self.future_prediction = 0

        self.crossed = False

        # upper and lower bound of etf
        self.etf_ub = self.etf_lb = 0

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
        if side == Side.BUY:
            q = self.etf_ub
            return np.min((self.constants.MAX_ORDER, self.constants.MAX_VOLUME - q))
        else:
            q = self.etf_lb
            return np.min((self.constants.MAX_ORDER, q + self.constants.MAX_VOLUME))


    def get_time(self):
        # not accounting for speed of the engine tho....
        elapsed_time = self.event_loop.time() - self.start_time
        elapsed_time *= self.speed
        return elapsed_time

    def insert(self, side, volume, price, lifespan=Lifespan.GOOD_FOR_DAY):
        
        if len(self.command_buffer) + 1 >= self.constants.MAX_MESSAGE:
            self.logger.info("Order not placed because of frequency limit")
            return

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

        price = int(price // 100) * 100

        self.send_insert_order(order_id, side, price, volume, lifespan)
        self.command_buffer.append(self.get_time())
        self.logger.info("(id: %d) Placing  %s %d lots for $%d", order_id, "bid" if side == Side.BUY else "ask", volume, price // 100)



    def cancel(self, order_id):
        if order_id > 0:
            if len(self.command_buffer) + 1 < self.constants.MAX_MESSAGE:
                self.send_cancel_order(order_id)
                self.command_buffer.append(self.get_time())
                self.logger.info("Cancelling %d", order_id)
            else:
                self.logger.info("Cancel not placed because of frequency limit")

    def round(self, x):
        return int(np.round(x / 100) * 100)

    @staticmethod
    def sigmoid(k, mu, x):
        return 1 / (1 + np.exp(-k * (x - mu)))


    def pricing(self, side):
        discount = - np.sign(self.etf_position) * np.abs(self.etf_position) * 6 / self.constants.MAX_VOLUME * 100

        etf = self.etf_orderbook.midpoint()
        etf_best_ask = self.etf_orderbook.best_ask()
        etf_best_bid = self.etf_orderbook.best_bid()
        future = self.future_orderbook.midpoint()
        future_best_bid = self.future_orderbook.best_bid()
        future_best_ask = self.future_orderbook.best_ask()

        if self.etf_orderbook.halfspread() > self.future_orderbook.halfspread():
            if future > etf:
                etf_best_bid = etf_best_ask - 2 * self.future_orderbook.halfspread()
            else:
                etf_best_ask = etf_best_bid + 2 * self.future_orderbook.halfspread()
        # gradient = self.future_orderbook.gradient(self.get_time()) / 2 

        if future > etf_best_ask + 200:
            etf_best_bid = etf
        elif future < etf_best_bid - 200:
            etf_best_ask = etf



        if future > etf:
            diff = future_best_bid + discount
        else:
            diff = future_best_ask + discount

        if diff > etf_best_ask:
            ask = diff
            bid = etf_best_bid

        elif etf_best_bid <= diff <= etf_best_ask:
            ask = etf_best_ask
            bid = etf_best_bid

        else:
            ask = etf_best_ask
            bid = diff

        # tie break
        bid = self.round(bid)
        ask = self.round(ask)

        if bid == ask:
            ask = ask + 100 

        # tie break
        bid = self.round(bid)
        ask = self.round(ask)

        if bid == ask:
            print(bid, ask)

        # ask
        if side == Side.SELL:
            return ask, Lifespan.GOOD_FOR_DAY

        # bid
        elif side == Side.BUY:
            return bid, Lifespan.GOOD_FOR_DAY

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

        """

        if np.sum(ask_volumes) + np.sum(bid_volumes) == 0:
            return

        if instrument == Instrument.FUTURE:
            self.future_orderbook.update_orders(bid_volumes, bid_prices, ask_volumes, ask_prices, self.get_time())
        elif instrument == Instrument.ETF:
            self.etf_orderbook.update_orders(bid_volumes, bid_prices, ask_volumes, ask_prices, self.get_time())

        if self.future_orderbook.total_volume == 0 or self.etf_orderbook.total_volume == 0:
            return

        # buy and sell volumes
        bid_volume = self.inventory(Side.BUY)
        ask_volume = self.inventory(Side.SELL)

        bid_price, bid_lifespan = self.pricing(Side.BUY)
        ask_price, ask_lifespan = self.pricing(Side.SELL)

        # if bid_price >= ask_price:
        #     bid_price = ask_price - 100

        crossed = (bid_price >= self.ask_price and self.ask_id != 0) or (ask_price <= self.bid_price and self.bid_id != 0)
        if crossed:
            print(bid_price, ask_price)

        # cancel all bids if changed or crossed
        if bid_price != self.bid_price or crossed:
            if self.bid_id != 0:
                self.cancel(self.bid_id)
            self.insert(Side.BUY, bid_volume, bid_price, bid_lifespan)

        # cancel all asks if changed or crossed
        if ask_price != self.ask_price or crossed:
            if self.ask_id != 0:
                self.cancel(self.ask_id)
            self.insert(Side.SELL, ask_volume, ask_price, ask_lifespan)

        self.ask_changed = self.bid_changed = False
        # print(self.orderbook.history)

        self.etf_ub = self.etf_position + self.bid_volume
        self.etf_lb = self.etf_position - self.ask_volume

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
                self.bid_price = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.ask_volume = 0
                self.ask_price = 0

            if fill_volume == 0:
                self.logger.info("Order %d cancelled", client_order_id)
            else:
                if client_order_id == self.bid_id:
                    self.etf_ub = self.etf_position + fill_volume
                elif client_order_id == self.ask_id:
                    self.etf_lb = self.etf_position - fill_volume
                self.logger.info("Order %d filled (%d / %d)", client_order_id, fill_volume, fill_volume + remaining_volume)
        elif fill_volume == 0:
            self.logger.info("Order %d placed", client_order_id)
        else:
            self.logger.info("Order %d filled (%d / %d)", client_order_id, fill_volume, fill_volume + remaining_volume)
            if client_order_id == self.bid_id:
                self.bid_changed = True
            elif client_order_id == self.ask_id:
                self.ask_changed = True


    def on_position_change_message(self, future_position: int, etf_position: int) -> None:
        """Called when your position changes.

        Since every trade in the ETF is automatically hedged in the future,
        future_position and etf_position will always be the inverse of each
        other (i.e. future_position == -1 * etf_position).
        """
        self.logger.info("Position changed to ETF: %d, FUTURE: %d", etf_position, future_position)
        self.etf_position = etf_position
        self.future_position = future_position

        self.etf_ub = self.etf_position + self.bid_volume
        self.etf_lb = self.etf_position - self.ask_volume


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

        # self.update_time = self.get_time()

        # if instrument == Instrument.ETF:
        #     self.orderbook.update(mean, self.update_time)
