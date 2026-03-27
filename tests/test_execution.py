"""
test_execution.py

Full unit-test suite for execution.py.
All MetaTrader5 calls are mocked via conftest.py + per-test patches.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import execution
from execution import (
    place_order,
    close_order,
    get_open_positions,
    count_open_trades,
    check_min_duration,
    close_all_positions,
)
import config

UTC = timezone.utc
MAGIC = config.BOT_MAGIC_NUMBER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tick(ask=1.10010, bid=1.10000):
    tick = MagicMock()
    tick.ask = ask
    tick.bid = bid
    return tick


def _make_order_result(retcode=10009, ticket=12345, price=1.10010, comment="done"):
    r = MagicMock()
    r.retcode  = retcode
    r.order    = ticket
    r.price    = price
    r.comment  = comment
    return r


def _make_position(
    ticket=12345,
    symbol="EURUSD",
    pos_type=0,          # 0=Buy, 1=Sell
    volume=0.10,
    price_open=1.10000,
    sl=1.09500,
    tp=1.10750,
    pos_time=None,
    profit=15.0,
    magic=MAGIC,
):
    p = MagicMock()
    p.ticket     = ticket
    p.symbol     = symbol
    p.type       = pos_type
    p.volume     = volume
    p.price_open = price_open
    p.sl         = sl
    p.tp         = tp
    p.time       = pos_time or int(datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp())
    p.profit     = profit
    p.magic      = magic
    return p


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    @patch("execution.time.sleep")          # suppress real sleeps
    @patch("execution.mt5")
    def test_buy_success(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = _make_order_result(
            retcode=10009, ticket=111, price=1.10010
        )
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY  = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL   = 1
        mock_mt5.ORDER_TIME_GTC      = 0
        mock_mt5.ORDER_FILLING_IOC   = 2

        result = place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)

        assert result is not None
        assert result["ticket"] == 111
        assert result["volume"] == 0.10
        assert result["sl"] == 1.09500
        assert result["tp"] == 1.10750
        # Buy → ask price used
        assert result["price"] == 1.10010

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_sell_uses_bid(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.return_value = _make_tick(ask=1.10010, bid=1.10000)
        mock_mt5.order_send.return_value = _make_order_result(
            retcode=10009, ticket=222, price=1.10000
        )
        mock_mt5.TRADE_RETCODE_DONE  = 10009
        mock_mt5.ORDER_TYPE_BUY      = 0
        mock_mt5.ORDER_TYPE_SELL     = 1
        mock_mt5.TRADE_ACTION_DEAL   = 1
        mock_mt5.ORDER_TIME_GTC      = 0
        mock_mt5.ORDER_FILLING_IOC   = 2

        result = place_order("EURUSD", -1, 0.10, 1.10750, 1.09500)

        assert result is not None
        assert result["price"] == 1.10000  # bid

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_retcode_failure_returns_none(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = _make_order_result(
            retcode=10019, ticket=0, price=0
        )
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY     = 0
        mock_mt5.ORDER_TYPE_SELL    = 1
        mock_mt5.TRADE_ACTION_DEAL  = 1
        mock_mt5.ORDER_TIME_GTC     = 0
        mock_mt5.ORDER_FILLING_IOC  = 2

        result = place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)
        assert result is None

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_tick_none_returns_none(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.return_value = None
        mock_mt5.last_error.return_value = (1, "no tick")

        result = place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)
        assert result is None
        mock_mt5.order_send.assert_not_called()

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_order_send_none_returns_none(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = None
        mock_mt5.last_error.return_value = (1, "connection lost")
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY     = 0
        mock_mt5.TRADE_ACTION_DEAL  = 1
        mock_mt5.ORDER_TIME_GTC     = 0
        mock_mt5.ORDER_FILLING_IOC  = 2

        result = place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)
        assert result is None

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_request_contains_magic_and_comment(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = _make_order_result()
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY     = 0
        mock_mt5.ORDER_TYPE_SELL    = 1
        mock_mt5.TRADE_ACTION_DEAL  = 1
        mock_mt5.ORDER_TIME_GTC     = 0
        mock_mt5.ORDER_FILLING_IOC  = 2

        place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)

        sent_req = mock_mt5.order_send.call_args[0][0]
        assert sent_req["magic"]   == config.BOT_MAGIC_NUMBER
        assert sent_req["comment"] == "ML_BOT_V1"
        assert sent_req["deviation"] == 10

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_exception_returns_none(self, mock_mt5, _sleep):
        mock_mt5.symbol_info_tick.side_effect = RuntimeError("MT5 crashed")

        result = place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)
        assert result is None


# ---------------------------------------------------------------------------
# 2-second order delay
# ---------------------------------------------------------------------------

class TestOrderDelay:
    def test_delay_enforced_between_orders(self):
        """_enforce_order_delay must sleep the remaining time."""
        execution._last_order_time = time.monotonic()   # simulate very recent order

        slept = []
        original_sleep = time.sleep

        with patch("execution.time.sleep", side_effect=lambda s: slept.append(s)):
            with patch("execution.mt5") as mock_mt5:
                mock_mt5.symbol_info_tick.return_value = _make_tick()
                mock_mt5.order_send.return_value = _make_order_result()
                mock_mt5.TRADE_RETCODE_DONE = 10009
                mock_mt5.ORDER_TYPE_BUY     = 0
                mock_mt5.ORDER_TYPE_SELL    = 1
                mock_mt5.TRADE_ACTION_DEAL  = 1
                mock_mt5.ORDER_TIME_GTC     = 0
                mock_mt5.ORDER_FILLING_IOC  = 2
                place_order("EURUSD", 1, 0.10, 1.09500, 1.10750)

        # A sleep of ~2 s must have been inserted
        assert any(s > 1.5 for s in slept), f"Expected ~2 s sleep, got {slept}"


# ---------------------------------------------------------------------------
# close_order
# ---------------------------------------------------------------------------

class TestCloseOrder:
    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_close_buy_position(self, mock_mt5, _sleep):
        pos = _make_position(ticket=999, pos_type=0)   # Buy
        mock_mt5.positions_get.return_value = [pos]
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = _make_order_result(retcode=10009)
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY     = 0
        mock_mt5.ORDER_TYPE_SELL    = 1
        mock_mt5.TRADE_ACTION_DEAL  = 1
        mock_mt5.ORDER_TIME_GTC     = 0
        mock_mt5.ORDER_FILLING_IOC  = 2

        assert close_order(999) is True
        # Closing a Buy → ORDER_TYPE_SELL used
        sent_req = mock_mt5.order_send.call_args[0][0]
        assert sent_req["type"] == 1   # ORDER_TYPE_SELL

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_close_sell_position(self, mock_mt5, _sleep):
        pos = _make_position(ticket=888, pos_type=1)   # Sell
        mock_mt5.positions_get.return_value = [pos]
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = _make_order_result(retcode=10009)
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY     = 0
        mock_mt5.ORDER_TYPE_SELL    = 1
        mock_mt5.TRADE_ACTION_DEAL  = 1
        mock_mt5.ORDER_TIME_GTC     = 0
        mock_mt5.ORDER_FILLING_IOC  = 2

        assert close_order(888) is True
        sent_req = mock_mt5.order_send.call_args[0][0]
        assert sent_req["type"] == 0   # ORDER_TYPE_BUY

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_ticket_not_found_returns_false(self, mock_mt5, _sleep):
        mock_mt5.positions_get.return_value = []

        assert close_order(404) is False
        mock_mt5.order_send.assert_not_called()

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_order_send_failure_returns_false(self, mock_mt5, _sleep):
        pos = _make_position(ticket=777, pos_type=0)
        mock_mt5.positions_get.return_value = [pos]
        mock_mt5.symbol_info_tick.return_value = _make_tick()
        mock_mt5.order_send.return_value = _make_order_result(retcode=10019)
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY     = 0
        mock_mt5.ORDER_TYPE_SELL    = 1
        mock_mt5.TRADE_ACTION_DEAL  = 1
        mock_mt5.ORDER_TIME_GTC     = 0
        mock_mt5.ORDER_FILLING_IOC  = 2
        mock_mt5.last_error.return_value = (1, "rejected")

        assert close_order(777) is False

    @patch("execution.time.sleep")
    @patch("execution.mt5")
    def test_exception_returns_false(self, mock_mt5, _sleep):
        mock_mt5.positions_get.side_effect = RuntimeError("MT5 crashed")

        assert close_order(123) is False


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------

class TestGetOpenPositions:
    @patch("execution.mt5")
    def test_filters_by_magic(self, mock_mt5):
        bot_pos   = _make_position(ticket=1, magic=MAGIC)
        other_pos = _make_position(ticket=2, magic=99999)
        mock_mt5.positions_get.return_value = [bot_pos, other_pos]
        mock_mt5.last_error.return_value = (0, "ok")

        result = get_open_positions()
        assert len(result) == 1
        assert result[0]["ticket"] == 1

    @patch("execution.mt5")
    def test_filters_by_symbol(self, mock_mt5):
        mock_mt5.positions_get.return_value = [
            _make_position(ticket=10, symbol="EURUSD", magic=MAGIC),
            _make_position(ticket=11, symbol="GBPUSD", magic=MAGIC),
        ]
        result = get_open_positions(symbol="EURUSD")
        # positions_get called with symbol=
        mock_mt5.positions_get.assert_called_once_with(symbol="EURUSD")

    @patch("execution.mt5")
    def test_returns_correct_keys(self, mock_mt5):
        mock_mt5.positions_get.return_value = [
            _make_position(ticket=5, magic=MAGIC)
        ]

        result = get_open_positions()
        keys = result[0].keys()
        for k in ("ticket", "symbol", "type", "volume", "open_price",
                  "sl", "tp", "open_time", "profit"):
            assert k in keys

    @patch("execution.mt5")
    def test_none_response_returns_empty(self, mock_mt5):
        mock_mt5.positions_get.return_value = None
        mock_mt5.last_error.return_value = (1, "error")
        assert get_open_positions() == []

    @patch("execution.mt5")
    def test_exception_returns_empty(self, mock_mt5):
        mock_mt5.positions_get.side_effect = RuntimeError("crash")
        assert get_open_positions() == []


# ---------------------------------------------------------------------------
# count_open_trades
# ---------------------------------------------------------------------------

class TestCountOpenTrades:
    @patch("execution.get_open_positions")
    def test_count_correct(self, mock_get):
        mock_get.return_value = [{"ticket": 1}, {"ticket": 2}]
        assert count_open_trades() == 2

    @patch("execution.get_open_positions")
    def test_passes_symbol(self, mock_get):
        mock_get.return_value = []
        count_open_trades(symbol="GBPUSD")
        mock_get.assert_called_once_with("GBPUSD")


# ---------------------------------------------------------------------------
# check_min_duration
# ---------------------------------------------------------------------------

class TestCheckMinDuration:
    @patch("execution.mt5")
    def test_position_old_enough(self, mock_mt5):
        old_time = int(
            (datetime.now(UTC).timestamp()) - config.MIN_TRADE_DURATION - 10
        )
        pos = _make_position(ticket=5, pos_time=old_time)
        mock_mt5.positions_get.return_value = [pos]

        assert check_min_duration(5) is True

    @patch("execution.mt5")
    def test_position_too_new(self, mock_mt5):
        new_time = int(datetime.now(UTC).timestamp()) - 5   # only 5 s old
        pos = _make_position(ticket=6, pos_time=new_time)
        mock_mt5.positions_get.return_value = [pos]

        assert check_min_duration(6) is False

    @patch("execution.mt5")
    def test_exactly_at_min_duration(self, mock_mt5):
        exact_time = int(
            datetime.now(UTC).timestamp() - config.MIN_TRADE_DURATION
        )
        pos = _make_position(ticket=7, pos_time=exact_time)
        mock_mt5.positions_get.return_value = [pos]

        assert check_min_duration(7) is True

    @patch("execution.mt5")
    def test_ticket_not_found_returns_false(self, mock_mt5):
        mock_mt5.positions_get.return_value = []
        assert check_min_duration(999) is False

    @patch("execution.mt5")
    def test_exception_returns_false(self, mock_mt5):
        mock_mt5.positions_get.side_effect = RuntimeError("crash")
        assert check_min_duration(1) is False


# ---------------------------------------------------------------------------
# close_all_positions
# ---------------------------------------------------------------------------

class TestCloseAllPositions:
    @patch("execution.time.sleep")
    @patch("execution.close_order")
    @patch("execution.get_open_positions")
    def test_closes_all_and_returns_count(self, mock_get, mock_close, _sleep):
        mock_get.return_value = [{"ticket": 1}, {"ticket": 2}]
        mock_close.return_value = True

        closed = close_all_positions()
        assert closed == 2
        assert mock_close.call_count == 2

    @patch("execution.time.sleep")
    @patch("execution.close_order")
    @patch("execution.get_open_positions")
    def test_2s_sleep_between_each_close(self, mock_get, mock_close, mock_sleep):
        mock_get.return_value = [{"ticket": 1}, {"ticket": 2}, {"ticket": 3}]
        mock_close.return_value = True

        close_all_positions()
        # One sleep(2) per position
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(2)

    @patch("execution.time.sleep")
    @patch("execution.close_order")
    @patch("execution.get_open_positions")
    def test_partial_failure_counted(self, mock_get, mock_close, _sleep):
        mock_get.return_value = [{"ticket": 1}, {"ticket": 2}]
        mock_close.side_effect = [True, False]   # second close fails

        closed = close_all_positions()
        assert closed == 1

    @patch("execution.get_open_positions")
    def test_no_positions_returns_zero(self, mock_get):
        mock_get.return_value = []
        assert close_all_positions() == 0
