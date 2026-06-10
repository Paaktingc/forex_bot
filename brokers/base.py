"""Shared broker adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BrokerAdapter(ABC):
    """Normalised live broker interface used by the trading bot."""

    name: str

    @abstractmethod
    def connect(self) -> float:
        """Connect to the broker and return account balance."""

    @abstractmethod
    def shutdown(self) -> None:
        """Close broker resources."""

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe_str: str, bars: int) -> pd.DataFrame:
        """Return UTC-indexed OHLCV bars."""

    @abstractmethod
    def get_latest_tick(self, symbol: str) -> dict[str, float]:
        """Return ask, bid, and spread in pips."""

    @abstractmethod
    def get_account_info(self) -> dict[str, float]:
        """Return account balance, equity, margin, free margin, and drawdown."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        signal: int,
        lot: float,
        sl_price: float,
        tp_price: float,
    ) -> Optional[dict]:
        """Place a market order. signal is 1 for buy and -1 for sell."""

    @abstractmethod
    def close_order(self, ticket: int) -> bool:
        """Close an open broker position."""

    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """Return open positions owned by this bot."""

    def count_open_trades(self, symbol: Optional[str] = None) -> int:
        return len(self.get_open_positions(symbol))

    @abstractmethod
    def check_min_duration(self, ticket: int) -> bool:
        """Return whether a position has been open for the configured minimum."""

    def close_all_positions(self) -> int:
        closed = 0
        for position in self.get_open_positions():
            if self.close_order(position["ticket"]):
                closed += 1
        return closed
