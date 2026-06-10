"""MetaTrader 5 broker adapter."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

import config
from .base import BrokerAdapter

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)
UTC = timezone.utc


class MT5BrokerAdapter(BrokerAdapter):
    name = "mt5"

    def __init__(self, mt5_module=None) -> None:
        self.mt5 = mt5_module if mt5_module is not None else mt5
        self._last_order_time: Optional[float] = None

    def _require_mt5(self) -> None:
        if self.mt5 is None or not MT5_AVAILABLE:
            raise RuntimeError("MT5 not available. Deploy to Windows VPS for live trading.")

    def _enforce_order_delay(self) -> None:
        if self._last_order_time is not None:
            elapsed = time.monotonic() - self._last_order_time
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)

    def _stamp_order_time(self) -> None:
        self._last_order_time = time.monotonic()

    def connect(self) -> float:
        self._require_mt5()
        load_dotenv()
        login_str = os.getenv("MT5_LOGIN") or os.getenv("MT5_ACCOUNT")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        if not (login_str and password and server):
            raise ConnectionError("Missing MT5 credentials in .env")

        login = int(login_str)
        if not self.mt5.initialize(login=login, password=password, server=server):
            raise ConnectionError(f"MT5 initialization failed: {self.mt5.last_error()}")

        account_info = self.mt5.account_info()
        if account_info is None:
            raise ConnectionError(f"Failed to get account info: {self.mt5.last_error()}")

        balance = float(account_info.balance)
        logger.info("Connected to MT5 | Account: %s | Balance: %.2f", login, balance)
        return balance

    def shutdown(self) -> None:
        if self.mt5 is not None and hasattr(self.mt5, "shutdown"):
            self.mt5.shutdown()

    def get_ohlcv(self, symbol: str, timeframe_str: str, bars: int) -> pd.DataFrame:
        self._require_mt5()
        timeframe_map = {
            "M15": self.mt5.TIMEFRAME_M15,
            "H1": self.mt5.TIMEFRAME_H1,
        }
        if timeframe_str not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}")

        rates = self.mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe_str], 0, bars)
        if rates is None or len(rates) == 0:
            logger.error("Failed to fetch rates for %s: %s", symbol, self.mt5.last_error())
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        if "tick_volume" in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        columns = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[columns].dropna()
        if "volume" in df.columns:
            df = df[df["volume"] > 0]
        return df

    def get_latest_tick(self, symbol: str) -> dict[str, float]:
        self._require_mt5()
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Failed to fetch tick for {symbol}: {self.mt5.last_error()}")
        ask = float(tick.ask)
        bid = float(tick.bid)
        return {"ask": ask, "bid": bid, "spread": (ask - bid) / 0.00001}

    def get_account_info(self) -> dict[str, float]:
        self._require_mt5()
        info = self.mt5.account_info()
        if info is None:
            raise ValueError(f"Failed to fetch account info: {self.mt5.last_error()}")
        balance = float(info.balance)
        equity = float(info.equity)
        margin = float(info.margin)
        free_margin = float(info.margin_free)
        drawdown_pct = ((balance - equity) / balance * 100.0) if balance > 0 else 0.0
        return {
            "balance": balance,
            "equity": equity,
            "margin": margin,
            "free_margin": free_margin,
            "drawdown_pct": drawdown_pct,
        }

    def place_order(
        self,
        symbol: str,
        signal: int,
        lot: float,
        sl_price: float,
        tp_price: float,
    ) -> Optional[dict]:
        self._require_mt5()
        self._enforce_order_delay()
        try:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error("place_order: cannot get tick for %s: %s", symbol, self.mt5.last_error())
                return None

            price = tick.ask if signal == 1 else tick.bid
            order_type = self.mt5.ORDER_TYPE_BUY if signal == 1 else self.mt5.ORDER_TYPE_SELL
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot),
                "type": order_type,
                "price": float(price),
                "sl": float(sl_price),
                "tp": float(tp_price),
                "deviation": 10,
                "magic": config.BOT_MAGIC_NUMBER,
                "comment": config.BOT_LABEL,
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            result = self.mt5.order_send(request)
            self._stamp_order_time()
            if result is None:
                logger.error("place_order: order_send returned None for %s: %s", symbol, self.mt5.last_error())
                return None
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(
                    "place_order: order rejected symbol=%s signal=%s lot=%s retcode=%s comment=%r error=%s",
                    symbol,
                    signal,
                    lot,
                    result.retcode,
                    getattr(result, "comment", ""),
                    self.mt5.last_error(),
                )
                return None
            return {
                "ticket": result.order,
                "price": result.price,
                "sl": sl_price,
                "tp": tp_price,
                "volume": float(lot),
                "time": datetime.now(UTC),
            }
        except Exception as exc:
            logger.exception("place_order: unexpected error for %s: %s", symbol, exc)
            return None

    def close_order(self, ticket: int) -> bool:
        self._require_mt5()
        self._enforce_order_delay()
        try:
            positions = self.mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error("close_order: no open position found for ticket %s.", ticket)
                return False

            pos = positions[0]
            tick = self.mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                logger.error("close_order: cannot get tick for %s: %s", pos.symbol, self.mt5.last_error())
                return False

            if pos.type == self.mt5.ORDER_TYPE_BUY:
                close_type = self.mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                close_type = self.mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": float(pos.volume),
                "type": close_type,
                "position": ticket,
                "price": float(price),
                "deviation": 10,
                "magic": config.BOT_MAGIC_NUMBER,
                "comment": f"{config.BOT_LABEL}_CLOSE",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            result = self.mt5.order_send(request)
            self._stamp_order_time()
            if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                retcode = result.retcode if result else "None"
                logger.error("close_order: failed ticket=%s retcode=%s error=%s", ticket, retcode, self.mt5.last_error())
                return False
            return True
        except Exception as exc:
            logger.exception("close_order: unexpected error for ticket %s: %s", ticket, exc)
            return False

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict]:
        self._require_mt5()
        try:
            raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
            if raw is None:
                logger.error("get_open_positions: positions_get failed: %s", self.mt5.last_error())
                return []
            result = []
            for pos in raw:
                if pos.magic != config.BOT_MAGIC_NUMBER:
                    continue
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": pos.type,
                    "volume": pos.volume,
                    "open_price": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "open_time": datetime.fromtimestamp(pos.time, tz=UTC),
                    "profit": pos.profit,
                })
            return result
        except Exception as exc:
            logger.exception("get_open_positions: unexpected error: %s", exc)
            return []

    def check_min_duration(self, ticket: int) -> bool:
        self._require_mt5()
        try:
            positions = self.mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning("check_min_duration: ticket %s not found.", ticket)
                return False
            open_time = datetime.fromtimestamp(positions[0].time, tz=UTC)
            elapsed = (datetime.now(UTC) - open_time).total_seconds()
            return elapsed >= config.MIN_TRADE_DURATION
        except Exception as exc:
            logger.exception("check_min_duration: unexpected error for ticket %s: %s", ticket, exc)
            return False

    def close_all_positions(self) -> int:
        positions = self.get_open_positions()
        closed = 0
        for pos in positions:
            if self.close_order(pos["ticket"]):
                closed += 1
            time.sleep(2)
        return closed
