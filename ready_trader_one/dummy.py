import asyncio

from typing import List, Tuple
from alec2 import Orderbook


from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side
import numpy as np
import itertools

class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.orderbook = Orderbook("ETF", True)

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error."""
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book."""

        if instrument == Instrument.ETF:
            self.orderbook.update(np.array(bid_volumes), np.array(bid_prices), np.array(ask_prices), np.array(ask_volumes))

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        """Called when the status of one of your orders changes."""
        pass
    def on_position_change_message(self, future_position: int, etf_position: int) -> None:
        """Called when your position changes."""
        pass