"""
test_risk_manager.py

Comprehensive unit tests for the RiskManager class.
Tests boundary conditions (at, just-under, just-over) for every check,
midnight reset of daily balance, martingale prevention, SL/TP maths,
and the can_trade gate.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_manager import RiskManager, is_rollover_window
import config

UTC = timezone.utc
BALANCE = 10_000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rm():
    """Fresh RiskManager with $10 000 starting balance."""
    return RiskManager(BALANCE)


# ---------------------------------------------------------------------------
# check_absolute_drawdown
# ---------------------------------------------------------------------------

class TestCheckAbsoluteDrawdown:
    # MAX_DRAWDOWN_PCT = 4.5 %  →  limit at 9 550.00
    _limit_equity = BALANCE * (1 - config.MAX_DRAWDOWN_PCT)   # 9 550.00

    def test_just_under_limit_allowed(self, rm):
        equity = self._limit_equity + 0.01   # 9 550.01 → 4.4999 %
        assert rm.check_absolute_drawdown(equity) is True

    def test_exactly_at_limit_halts(self, rm):
        with patch("execution.close_all_positions") as mock_close:
            assert rm.check_absolute_drawdown(self._limit_equity) is False
            mock_close.assert_called_once()

    def test_over_limit_halts(self, rm):
        with patch("execution.close_all_positions") as mock_close:
            equity = self._limit_equity - 100
            assert rm.check_absolute_drawdown(equity) is False
            mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# check_daily_loss
# ---------------------------------------------------------------------------

class TestCheckDailyLoss:
    # DAILY_LOSS_PCT = 4.0 %  →  limit at 9 600.00
    _limit_equity = BALANCE * (1 - config.DAILY_LOSS_PCT)   # 9 600.00

    def test_just_under_limit_allowed(self, rm):
        equity = self._limit_equity + 0.01   # 9 600.01
        assert rm.check_daily_loss(equity) is True

    def test_exactly_at_limit_halts(self, rm):
        assert rm.check_daily_loss(self._limit_equity) is False
        assert rm.halted_today is True

    def test_over_limit_halts(self, rm):
        assert rm.check_daily_loss(self._limit_equity - 100) is False
        assert rm.halted_today is True

    def test_subsequent_calls_blocked_when_halted(self, rm):
        rm.halted_today = True
        # Even if equity recovers, trading stays blocked for the day
        assert rm.check_daily_loss(BALANCE + 500) is False

    def test_same_day_does_not_clear_halt(self, rm):
        rm.halted_today = True
        assert rm.check_daily_loss(BALANCE + 500) is False
        assert rm.halted_today is True

    def test_new_utc_day_clears_halt_on_can_trade(self, rm):
        rm.halted_today = True
        rm.daily_start_time = datetime.now(UTC) - timedelta(days=1)

        with patch("risk_manager.is_rollover_window", return_value=False):
            ok, msg = rm.can_trade(BALANCE, 0)

        assert ok is True
        assert msg == "OK"
        assert rm.halted_today is False

    def test_midnight_reset_updates_start_time(self, rm):
        old_date = datetime.now(UTC).date() - timedelta(days=1)
        rm.daily_start_time = datetime(old_date.year, old_date.month, old_date.day,
                                       tzinfo=UTC)
        rm._maybe_reset_daily(BALANCE)
        assert rm.daily_start_time.date() == datetime.now(UTC).date()

    def test_new_day_start_balance_anchors_to_current_equity(self, rm):
        rm.halted_today = True
        rm.daily_start_balance = BALANCE
        rm.daily_start_time = datetime.now(UTC) - timedelta(days=1)

        assert rm.check_daily_loss(9_500.0) is True
        assert rm.halted_today is False
        assert rm.daily_start_balance == 9_500.0


# ---------------------------------------------------------------------------
# calculate_lot_size
# ---------------------------------------------------------------------------

class TestCalculateLotSize:
    def test_standard_calculation(self, rm):
        # equity=10000, risk=39 (0.39%), sl_pips=50, pip_val=10 -> lot=0.078 -> 0.07
        lot = rm.calculate_lot_size(10_000, 1.0950, 1.1000, "EURUSD")
        assert lot == 0.07

    def test_lot_size_floors_to_broker_step(self, rm):
        # raw lot is about 0.147, so conservative step rounding floors to 0.14.
        lot = rm.calculate_lot_size(10_000, 1.09735, 1.1000, "EURUSD")
        assert lot == 0.14

    def test_entry_equals_sl_returns_zero(self, rm):
        lot = rm.calculate_lot_size(10_000, 1.1000, 1.1000, "EURUSD")
        assert lot == 0.0

    def test_no_forced_minimum_lot_when_risk_size_too_small(self):
        # Tiny equity / wide SL -> lot would be near-zero, so block the trade.
        # Use a matching starting balance so the drawdown circuit breaker
        # (which keys off equity vs. starting balance) stays inactive.
        rm = RiskManager(10)
        lot = rm.calculate_lot_size(10, 1.0000, 1.5000, "EURUSD")
        assert lot == 0.0

    def test_maximum_clamp(self, rm):
        # Massive equity / tiny SL → lot > 5.0 → clamped to 5.0
        lot = rm.calculate_lot_size(10_000_000, 1.0999, 1.1000, "EURUSD")
        assert lot == 5.0

    def test_unsupported_symbol_returns_zero(self, rm):
        lot = rm.calculate_lot_size(10_000, 1.0950, 1.1000, "GBPUSD")
        assert lot == 0.0

    def test_martingale_prevention(self):
        # Keep the equity decline within the circuit-breaker warning threshold
        # so this test isolates the martingale cap, not the drawdown breaker.
        rm = RiskManager(9_600)
        rm._last_lot_size = 0.05
        rm._last_equity_at_lot = 9_600.0   # previous equity was higher

        # New equity is lower (≈1% DD), so we calculate a size – but it must
        # not exceed the previous lot of 0.05.
        lot = rm.calculate_lot_size(9_500, 1.0950, 1.1000, "EURUSD")
        # uncapped ≈ 0.07 > previous 0.05 → capped to 0.05
        assert lot <= 0.05

    def test_no_martingale_cap_on_profit(self, rm):
        # Equity grew since last trade → no cap applied
        rm._last_lot_size = 0.05
        rm._last_equity_at_lot = 9_000.0   # previous equity was lower

        lot = rm.calculate_lot_size(10_000, 1.0950, 1.1000, "EURUSD")
        # uncapped floors to 0.07; previous lot was 0.05 (smaller), so no cap.
        assert lot == 0.07


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------

class TestDrawdownCircuitBreaker:
    _entry = 1.1000
    _sl = 1.0950   # 50 pips

    def test_full_size_below_warning(self, rm):
        # 2.9% drawdown → below 3% warning → full size, multiplier 1.0
        equity = BALANCE * (1 - 0.029)
        assert rm._drawdown_size_multiplier(equity) == 1.0

    def test_half_size_at_warning(self, rm):
        # Exactly 3.0% drawdown → halve position size
        equity = BALANCE * (1 - config.DRAWDOWN_WARNING_THRESHOLD)
        assert rm._drawdown_size_multiplier(equity) == 0.5

        full = rm.calculate_lot_size(BALANCE, self._sl, self._entry, "EURUSD")
        rm._last_lot_size = 0.0   # clear martingale state between calls
        reduced = rm.calculate_lot_size(equity, self._sl, self._entry, "EURUSD")
        assert reduced < full

    def test_halt_at_hard_limit(self, rm):
        # Exactly 4.5% drawdown → sizing blocked, returns 0
        equity = BALANCE * (1 - config.MAX_DRAWDOWN_LIMIT)
        assert rm._drawdown_size_multiplier(equity) == 0.0
        lot = rm.calculate_lot_size(equity, self._sl, self._entry, "EURUSD")
        assert lot == 0.0

    def test_no_drawdown_in_profit(self, rm):
        # Equity above starting balance → zero drawdown, full size
        assert rm.current_drawdown_pct(BALANCE + 500) == 0.0
        assert rm._drawdown_size_multiplier(BALANCE + 500) == 1.0


# ---------------------------------------------------------------------------
# calculate_sl_tp
# ---------------------------------------------------------------------------

class TestCalculateSlTp:
    # ATR = 0.0010, SL_MULT=1.0, TP_MULT=1.5
    _atr = 0.0010
    _entry = 1.10000

    def test_buy_sl_below_tp_above(self, rm):
        sl, tp = rm.calculate_sl_tp(1, self._entry, self._atr)
        expected_sl = round(self._entry - config.SL_ATR_MULT * self._atr, 5)
        expected_tp = round(self._entry + config.TP_ATR_MULT * self._atr, 5)
        assert sl == expected_sl
        assert tp == expected_tp
        assert sl < self._entry < tp

    def test_sell_sl_above_tp_below(self, rm):
        sl, tp = rm.calculate_sl_tp(-1, self._entry, self._atr)
        expected_sl = round(self._entry + config.SL_ATR_MULT * self._atr, 5)
        expected_tp = round(self._entry - config.TP_ATR_MULT * self._atr, 5)
        assert sl == expected_sl
        assert tp == expected_tp
        assert tp < self._entry < sl

    def test_rounding_to_5dp(self, rm):
        sl, tp = rm.calculate_sl_tp(1, 1.123456789, 0.0003333)
        assert len(str(sl).split(".")[-1]) <= 5
        assert len(str(tp).split(".")[-1]) <= 5


# ---------------------------------------------------------------------------
# can_trade
# ---------------------------------------------------------------------------

class TestCanTrade:
    def test_all_clear(self, rm):
        with patch("risk_manager.is_rollover_window", return_value=False):
            ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is True
        assert msg == "OK"

    def test_drawdown_breach(self, rm):
        equity = BALANCE * (1 - config.MAX_DRAWDOWN_PCT) - 1
        with patch("execution.close_all_positions"):
            ok, msg = rm.can_trade(equity, 0)
        assert ok is False
        assert msg == "MAX DRAWDOWN HIT"

    def test_daily_loss_breach(self, rm):
        equity = BALANCE * (1 - config.DAILY_LOSS_PCT) - 1
        with patch("risk_manager.is_rollover_window", return_value=False):
            ok, msg = rm.can_trade(equity, 0)
        assert ok is False
        assert msg == "DAILY LOSS LIMIT HIT"

    def test_rollover_window(self, rm):
        with patch("risk_manager.is_rollover_window", return_value=True):
            ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is False
        assert msg == "ROLLOVER WINDOW"

    def test_max_trades_open(self, rm):
        with patch("risk_manager.is_rollover_window", return_value=False):
            ok, msg = rm.can_trade(BALANCE, config.MAX_OPEN_TRADES)
        assert ok is False
        assert msg == "MAX TRADES OPEN"

    def test_priority_drawdown_before_daily(self, rm):
        """Absolute drawdown takes priority over daily loss."""
        equity = BALANCE * (1 - config.MAX_DRAWDOWN_PCT) - 1
        with patch("execution.close_all_positions"):
            ok, msg = rm.can_trade(equity, 0)
        assert msg == "MAX DRAWDOWN HIT"


# ---------------------------------------------------------------------------
# is_rollover_window
# ---------------------------------------------------------------------------

class TestIsRolloverWindow:
    def test_inside_window(self):
        inside_hour = config.ROLLOVER_START_UTC
        fake_now = datetime(2023, 1, 1, inside_hour, 30, tzinfo=UTC)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            assert is_rollover_window() is True

    def test_outside_window_before(self):
        outside_hour = config.ROLLOVER_START_UTC - 1
        fake_now = datetime(2023, 1, 1, outside_hour, 59, tzinfo=UTC)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            assert is_rollover_window() is False

    def test_outside_window_after(self):
        outside_hour = config.ROLLOVER_END_UTC
        fake_now = datetime(2023, 1, 1, outside_hour, 0, tzinfo=UTC)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            assert is_rollover_window() is False
