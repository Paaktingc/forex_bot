"""cTrader Open API broker adapter.

This adapter is intentionally imported only when BROKER=ctrader so local MT5
tests and offline workflows do not need the cTrader SDK installed.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

import config
from .base import BrokerAdapter

logger = logging.getLogger(__name__)
UTC = timezone.utc


def _load_sdk() -> dict[str, Any]:
    try:
        from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAAccountAuthReq,
            ProtoOAApplicationAuthReq,
            ProtoOAClosePositionReq,
            ProtoOAGetAccountListByAccessTokenReq,
            ProtoOAGetTrendbarsReq,
            ProtoOANewOrderReq,
            ProtoOAReconcileReq,
            ProtoOARefreshTokenReq,
            ProtoOASpotEvent,
            ProtoOASubscribeSpotsReq,
            ProtoOASymbolsListReq,
            ProtoOATraderReq,
        )
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
            ProtoOAOrderType,
            ProtoOATimeInForce,
            ProtoOATradeSide,
            ProtoOATrendbarPeriod,
        )
        from twisted.internet import reactor
    except ImportError as exc:
        raise RuntimeError(
            "cTrader support requires `pip install ctrader-open-api`. "
            "Install requirements and set BROKER=ctrader."
        ) from exc

    return {
        "Client": Client,
        "EndPoints": EndPoints,
        "Protobuf": Protobuf,
        "TcpProtocol": TcpProtocol,
        "ProtoOAAccountAuthReq": ProtoOAAccountAuthReq,
        "ProtoOAApplicationAuthReq": ProtoOAApplicationAuthReq,
        "ProtoOAClosePositionReq": ProtoOAClosePositionReq,
        "ProtoOAGetAccountListByAccessTokenReq": ProtoOAGetAccountListByAccessTokenReq,
        "ProtoOAGetTrendbarsReq": ProtoOAGetTrendbarsReq,
        "ProtoOANewOrderReq": ProtoOANewOrderReq,
        "ProtoOAReconcileReq": ProtoOAReconcileReq,
        "ProtoOARefreshTokenReq": ProtoOARefreshTokenReq,
        "ProtoOASpotEvent": ProtoOASpotEvent,
        "ProtoOASubscribeSpotsReq": ProtoOASubscribeSpotsReq,
        "ProtoOASymbolsListReq": ProtoOASymbolsListReq,
        "ProtoOATraderReq": ProtoOATraderReq,
        "ProtoOAOrderType": ProtoOAOrderType,
        "ProtoOATimeInForce": ProtoOATimeInForce,
        "ProtoOATradeSide": ProtoOATradeSide,
        "ProtoOATrendbarPeriod": ProtoOATrendbarPeriod,
        "reactor": reactor,
    }


class CTraderBrokerAdapter(BrokerAdapter):
    name = "ctrader"

    def __init__(self) -> None:
        self.sdk: dict[str, Any] | None = None
        self.client: Any = None
        self.account_id: int | None = None
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self._symbols_by_name: dict[str, Any] = {}
        self._spots_by_symbol_id: dict[int, Any] = {}
        self._last_order_time: Optional[float] = None

    def connect(self) -> float:
        load_dotenv()
        self.sdk = _load_sdk()
        self.access_token = os.getenv("CTRADER_ACCESS_TOKEN")
        self.refresh_token = os.getenv("CTRADER_REFRESH_TOKEN")
        client_id = os.getenv("CTRADER_CLIENT_ID")
        client_secret = os.getenv("CTRADER_CLIENT_SECRET")
        account_id = os.getenv("CTRADER_ACCOUNT_ID")
        environment = os.getenv("CTRADER_ENV", config.CTRADER_ENV).lower()

        if not (client_id and client_secret and self.access_token and account_id):
            raise ConnectionError(
                "Missing cTrader credentials. Set CTRADER_CLIENT_ID, "
                "CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, and CTRADER_ACCOUNT_ID."
            )

        self.account_id = int(account_id)
        self._ensure_reactor()
        endpoints = self.sdk["EndPoints"]
        host = endpoints.PROTOBUF_LIVE_HOST if environment == "live" else endpoints.PROTOBUF_DEMO_HOST
        self.client = self.sdk["Client"](host, endpoints.PROTOBUF_PORT, self.sdk["TcpProtocol"])
        self._start_client_service()

        self._send(self.sdk["ProtoOAApplicationAuthReq"](
            clientId=client_id,
            clientSecret=client_secret,
        ))

        if self.refresh_token:
            try:
                refreshed = self._send(self.sdk["ProtoOARefreshTokenReq"](
                    refreshToken=self.refresh_token,
                    clientId=client_id,
                    clientSecret=client_secret,
                ))
                self.access_token = getattr(refreshed, "accessToken", self.access_token)
                self.refresh_token = getattr(refreshed, "refreshToken", self.refresh_token)
            except Exception:
                logger.warning("cTrader token refresh failed; trying supplied access token.")

        self._send(self.sdk["ProtoOAAccountAuthReq"](
            ctidTraderAccountId=self.account_id,
            accessToken=self.access_token,
        ))
        self._load_symbols()
        account = self.get_account_info()
        logger.info(
            "Connected to cTrader %s | Account: %s | Balance: %.2f",
            environment,
            self.account_id,
            account["balance"],
        )
        return account["balance"]

    def shutdown(self) -> None:
        self.client = None

    def get_ohlcv(self, symbol: str, timeframe_str: str, bars: int) -> pd.DataFrame:
        self._require_connected()
        spec = self._symbol_spec(symbol)
        period = self._period(timeframe_str)
        minutes = 15 if timeframe_str == "M15" else 60
        to_ts = datetime.now(UTC)
        from_ts = to_ts - timedelta(minutes=minutes * max(bars + 5, 10))
        request = self.sdk["ProtoOAGetTrendbarsReq"](
            ctidTraderAccountId=self.account_id,
            symbolId=spec.symbolId,
            period=period,
            fromTimestamp=int(from_ts.timestamp() * 1000),
            toTimestamp=int(to_ts.timestamp() * 1000),
            count=bars,
        )
        response = self._send(request)
        rows = []
        digits = int(getattr(spec, "digits", 5))
        for bar in getattr(response, "trendbar", []):
            low_raw = int(getattr(bar, "low"))
            timestamp = getattr(bar, "utcTimestampInMinutes", None)
            if timestamp is None:
                timestamp = getattr(bar, "timestamp", 0) // 60000
            rows.append({
                "time": pd.to_datetime(int(timestamp), unit="m", utc=True),
                "open": self._price(low_raw + int(getattr(bar, "deltaOpen", 0)), digits),
                "high": self._price(low_raw + int(getattr(bar, "deltaHigh", 0)), digits),
                "low": self._price(low_raw, digits),
                "close": self._price(low_raw + int(getattr(bar, "deltaClose", 0)), digits),
                "volume": float(getattr(bar, "volume", 0)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        # Drop the still-forming trendbar so callers only ever see CLOSED bars
        # (Finding 6). strategy.generate_candidate assumes iloc[-1] is closed.
        from bar_utils import drop_forming_bar

        return drop_forming_bar(df, minutes)

    def get_latest_tick(self, symbol: str) -> dict[str, float]:
        self._require_connected()
        spec = self._symbol_spec(symbol)
        self._send(self.sdk["ProtoOASubscribeSpotsReq"](
            ctidTraderAccountId=self.account_id,
            symbolId=[spec.symbolId],
        ))
        event = self._wait_for_spot(int(spec.symbolId))
        digits = int(getattr(spec, "digits", 5))
        ask = self._spot_price(int(getattr(event, "ask")))
        bid = self._spot_price(int(getattr(event, "bid")))
        pip_size = 10 ** -int(getattr(spec, "pipPosition", 4))
        return {"ask": ask, "bid": bid, "spread": (ask - bid) / pip_size}

    def get_account_info(self) -> dict[str, float]:
        self._require_connected()
        trader = self._send(self.sdk["ProtoOATraderReq"](
            ctidTraderAccountId=self.account_id,
        ))
        account = getattr(trader, "trader", trader)
        money_digits = int(getattr(account, "moneyDigits", 2))
        balance_raw = getattr(account, "balance", 0)
        equity_raw = getattr(account, "equity", balance_raw)
        margin_raw = getattr(account, "margin", 0)
        free_margin_raw = getattr(account, "freeMargin", int(equity_raw) - int(margin_raw))
        balance = self._money(balance_raw, money_digits)
        equity = self._money(equity_raw, money_digits)
        margin = self._money(margin_raw, money_digits)
        free_margin = self._money(free_margin_raw, money_digits)
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
        self._require_connected()
        self._enforce_order_delay()
        spec = self._symbol_spec(symbol)
        try:
            tick = self.get_latest_tick(symbol)
            entry_price = tick["ask"] if signal == 1 else tick["bid"]
            order_type = self._enum(self.sdk["ProtoOAOrderType"], "MARKET", 1)
            trade_side = self._enum(self.sdk["ProtoOATradeSide"], "BUY" if signal == 1 else "SELL", 1 if signal == 1 else 2)
            time_in_force = self._enum(self.sdk["ProtoOATimeInForce"], "IMMEDIATE_OR_CANCEL", 3)
            relative_sl = int(round(abs(entry_price - float(sl_price)) * 100_000))
            relative_tp = int(round(abs(float(tp_price) - entry_price) * 100_000))
            request = self.sdk["ProtoOANewOrderReq"](
                ctidTraderAccountId=self.account_id,
                symbolId=spec.symbolId,
                orderType=order_type,
                tradeSide=trade_side,
                volume=self._lot_to_protocol_volume(lot),
                timeInForce=time_in_force,
                relativeStopLoss=relative_sl,
                relativeTakeProfit=relative_tp,
                comment=config.BOT_LABEL,
                label=config.BOT_LABEL,
            )
            response = self._send(request)
            self._stamp_order_time()
            if getattr(response, "errorCode", ""):
                logger.error("cTrader order rejected: %s", response.errorCode)
                return None
            position = getattr(response, "position", None)
            deal = getattr(response, "deal", None)
            ticket = getattr(position, "positionId", None) or getattr(deal, "positionId", None)
            price = getattr(deal, "executionPrice", None) or entry_price
            return {
                "ticket": int(ticket) if ticket is not None else 0,
                "price": float(price),
                "sl": sl_price,
                "tp": tp_price,
                "volume": float(lot),
                "time": datetime.now(UTC),
            }
        except Exception as exc:
            logger.exception("cTrader place_order failed for %s: %s", symbol, exc)
            return None

    def close_order(self, ticket: int) -> bool:
        self._require_connected()
        self._enforce_order_delay()
        position = next((p for p in self.get_open_positions() if int(p["ticket"]) == int(ticket)), None)
        if position is None:
            logger.error("cTrader close_order: position %s not found.", ticket)
            return False
        response = self._send(self.sdk["ProtoOAClosePositionReq"](
            ctidTraderAccountId=self.account_id,
            positionId=int(ticket),
            volume=self._lot_to_protocol_volume(float(position["volume"])),
        ))
        self._stamp_order_time()
        return not bool(getattr(response, "errorCode", ""))

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict]:
        self._require_connected()
        response = self._send(self.sdk["ProtoOAReconcileReq"](
            ctidTraderAccountId=self.account_id,
        ))
        positions = []
        for pos in getattr(response, "position", []):
            trade_data = getattr(pos, "tradeData", None)
            symbol_id = getattr(trade_data, "symbolId", None)
            spec = self._symbol_by_id(symbol_id)
            symbol_name = getattr(spec, "symbolName", str(symbol_id))
            if symbol and symbol_name != symbol:
                continue
            label = getattr(trade_data, "label", "") or getattr(trade_data, "comment", "")
            if label and config.BOT_LABEL not in label:
                continue
            positions.append({
                "ticket": getattr(pos, "positionId"),
                "symbol": symbol_name,
                "type": 0 if getattr(trade_data, "tradeSide", 1) == self._enum(self.sdk["ProtoOATradeSide"], "BUY", 1) else 1,
                "volume": self._protocol_volume_to_lot(getattr(trade_data, "volume", 0)),
                "open_price": float(getattr(pos, "price", 0.0)),
                "sl": float(getattr(pos, "stopLoss", 0.0)),
                "tp": float(getattr(pos, "takeProfit", 0.0)),
                "open_time": datetime.fromtimestamp(
                    int(getattr(trade_data, "openTimestamp", 0)) / 1000,
                    tz=UTC,
                ),
                "profit": self._money(getattr(pos, "profit", 0), int(getattr(pos, "moneyDigits", 2))),
            })
        return positions

    def check_min_duration(self, ticket: int) -> bool:
        position = next((p for p in self.get_open_positions() if int(p["ticket"]) == int(ticket)), None)
        if not position:
            return False
        elapsed = (datetime.now(UTC) - position["open_time"]).total_seconds()
        return elapsed >= config.MIN_TRADE_DURATION

    def close_all_positions(self) -> int:
        closed = 0
        for pos in self.get_open_positions():
            if self.close_order(pos["ticket"]):
                closed += 1
            time.sleep(2)
        return closed

    def _ensure_reactor(self) -> None:
        reactor = self.sdk["reactor"]
        if reactor.running:
            return
        thread = threading.Thread(target=reactor.run, kwargs={"installSignalHandlers": False}, daemon=True)
        thread.start()
        timeout = time.monotonic() + 5
        while not reactor.running and time.monotonic() < timeout:
            time.sleep(0.01)
        if not reactor.running:
            raise ConnectionError("Timed out starting Twisted reactor for cTrader.")

    def _start_client_service(self) -> None:
        connected = threading.Event()

        def _connected(_client: Any) -> None:
            connected.set()

        def _message_received(_client: Any, message: Any) -> None:
            try:
                parsed = self.sdk["Protobuf"].extract(message)
            except Exception:
                parsed = message
            if isinstance(parsed, self.sdk["ProtoOASpotEvent"]):
                self._spots_by_symbol_id[int(parsed.symbolId)] = parsed

        if hasattr(self.client, "setConnectedCallback"):
            self.client.setConnectedCallback(_connected)
        if hasattr(self.client, "setMessageReceivedCallback"):
            self.client.setMessageReceivedCallback(_message_received)
        if hasattr(self.client, "startService"):
            self.client.startService()
        if not connected.wait(10):
            logger.warning("Timed out waiting for cTrader connected callback; continuing with first request.")

    def _send(self, request: Any, timeout: float = 15.0) -> Any:
        self._require_client()
        event = threading.Event()
        box: dict[str, Any] = {}

        def _ok(response: Any) -> Any:
            box["response"] = response
            event.set()
            return response

        def _err(failure: Any) -> Any:
            box["error"] = failure
            event.set()
            return failure

        def _do_send() -> None:
            try:
                deferred = self.client.send(request)
                deferred.addCallbacks(_ok, _err)
            except Exception as exc:
                box["error"] = exc
                event.set()

        self.sdk["reactor"].callFromThread(_do_send)
        if not event.wait(timeout):
            raise TimeoutError(f"Timed out waiting for cTrader response to {type(request).__name__}")
        if "error" in box:
            failure = box["error"]
            if isinstance(failure, Exception):
                raise failure
            raise RuntimeError(getattr(failure, "getErrorMessage", lambda: str(failure))())
        return box["response"]

    def _require_connected(self) -> None:
        if self.client is None or self.account_id is None or self.sdk is None:
            raise ConnectionError("cTrader broker is not connected. Call connect() first.")

    def _require_client(self) -> None:
        if self.client is None:
            raise ConnectionError("cTrader client is not initialized.")

    def _load_symbols(self) -> None:
        response = self._send(self.sdk["ProtoOASymbolsListReq"](
            ctidTraderAccountId=self.account_id,
        ))
        self._symbols_by_name = {
            getattr(symbol, "symbolName"): symbol
            for symbol in getattr(response, "symbol", [])
            if getattr(symbol, "symbolName", None)
        }

    def _symbol_spec(self, symbol: str) -> Any:
        if symbol not in self._symbols_by_name:
            self._load_symbols()
        if symbol not in self._symbols_by_name:
            raise ValueError(f"Symbol {symbol!r} not found in cTrader account symbols.")
        return self._symbols_by_name[symbol]

    def _symbol_by_id(self, symbol_id: int) -> Any:
        for spec in self._symbols_by_name.values():
            if int(getattr(spec, "symbolId", -1)) == int(symbol_id):
                return spec
        return type("UnknownSymbol", (), {"symbolName": str(symbol_id), "digits": 5, "pipPosition": 4})()

    def _wait_for_spot(self, symbol_id: int, timeout: float = 5.0) -> Any:
        cached = self._spots_by_symbol_id.get(symbol_id)
        if cached is not None:
            return cached
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.05)
            cached = self._spots_by_symbol_id.get(symbol_id)
            if cached is not None:
                return cached
        raise TimeoutError(f"Timed out waiting for cTrader spot update for symbolId={symbol_id}")

    def _period(self, timeframe_str: str) -> int:
        enum = self.sdk["ProtoOATrendbarPeriod"]
        name = {"M15": "M15", "H1": "H1"}.get(timeframe_str)
        if name is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}")
        return self._enum(enum, name, 4 if timeframe_str == "M15" else 7)

    @staticmethod
    def _enum(enum_obj: Any, name: str, fallback: int) -> int:
        return int(getattr(enum_obj, name, fallback))

    @staticmethod
    def _price(relative_price: int, digits: int) -> float:
        return float(relative_price) / (10 ** digits)

    @staticmethod
    def _spot_price(relative_price: int) -> float:
        return float(relative_price) / 100_000.0

    @staticmethod
    def _money(value: Any, digits: int = 2) -> float:
        return float(value) / (10 ** digits)

    @staticmethod
    def _lot_to_protocol_volume(lot: float) -> int:
        units = float(lot) * config.CTRADER_UNITS_PER_LOT
        return int(round(units * 100))

    @staticmethod
    def _protocol_volume_to_lot(volume: int) -> float:
        return float(volume) / 100.0 / config.CTRADER_UNITS_PER_LOT

    def _enforce_order_delay(self) -> None:
        if self._last_order_time is not None:
            elapsed = time.monotonic() - self._last_order_time
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)

    def _stamp_order_time(self) -> None:
        self._last_order_time = time.monotonic()
