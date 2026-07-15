"""
test_risk_manager.py

Unit tests for the The5ers Bootcamp RiskManager.
Covers the −3% kill switch (flatten + persisted disable), the official −5%
backstop, weekly stop (−1.5% / 5 consecutive losses), daily stop (−0.75% /
2 trades / 2 consecutive losses), server-midnight and Monday resets,
SL clamp skip, TP = 2R math, lot round-down, heartbeat, and the can_trade gate.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_manager import RiskManager, is_rollover_window, is_no_trade_server_window
import config

UTC = timezone.utc
BALANCE = 10_000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rm(tmp_path):
    """Fresh RiskManager with $10 000 starting balance and isolated flag file."""
    return RiskManager(BALANCE, disabled_flag_path=tmp_path / "disabled.json")


@pytest.fixture(autouse=True)
def _quiet_windows():
    """Keep can_trade deterministic regardless of wall-clock time."""
    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        yield


# ---------------------------------------------------------------------------
# Kill switch (−3%, operative halt)
# ---------------------------------------------------------------------------

class TestKillSwitch:
    _limit_equity = BALANCE * (1 - config.KILL_SWITCH_PCT)   # 9 700.00

    def test_just_under_limit_allowed(self, rm):
        assert rm.check_kill_switch(self._limit_equity + 0.01) is True
        assert not rm.is_disabled()

    def test_at_limit_flattens_and_disables(self, rm):
        with patch("execution.close_all_positions") as mock_close:
            assert rm.check_kill_switch(self._limit_equity) is False
            mock_close.assert_called_once()
        assert rm.is_disabled()

    def test_disabled_flag_persists_across_restart(self, rm, tmp_path):
        with patch("execution.close_all_positions"):
            rm.check_kill_switch(self._limit_equity - 50)
        assert rm.is_disabled()

        # Simulate a restart: brand-new RiskManager, same flag path
        rm2 = RiskManager(BALANCE, disabled_flag_path=tmp_path / "disabled.json")
        assert rm2.is_disabled()
        ok, msg = rm2.can_trade(BALANCE, 0)
        assert ok is False
        assert "DISABLED" in msg

    def test_manual_re_arm_by_deleting_flag(self, rm, tmp_path):
        with patch("execution.close_all_positions"):
            rm.check_kill_switch(self._limit_equity)
        (tmp_path / "disabled.json").unlink()
        assert not rm.is_disabled()
        ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is True

    def test_kill_switch_fires_before_official_limit(self, rm):
        """The −5% official limit must never be the working limit."""
        equity = BALANCE * (1 - config.KILL_SWITCH_PCT) - 1  # −3% and a bit
        assert equity > BALANCE * (1 - config.MAX_DRAWDOWN_LIMIT)
        with patch("execution.close_all_positions"):
            ok, msg = rm.can_trade(equity, 0)
        assert ok is False
        assert msg == "KILL SWITCH HIT"


# ---------------------------------------------------------------------------
# Official −5% backstop
# ---------------------------------------------------------------------------

class TestCheckAbsoluteDrawdown:
    _limit_equity = BALANCE * (1 - config.MAX_DRAWDOWN_LIMIT)   # 9 500.00

    def test_just_under_limit_allowed(self, rm):
        assert rm.check_absolute_drawdown(self._limit_equity + 0.01) is True

    def test_exactly_at_limit_halts(self, rm):
        with patch("execution.close_all_positions") as mock_close:
            assert rm.check_absolute_drawdown(self._limit_equity) is False
            mock_close.assert_called_once()

    def test_over_limit_halts(self, rm):
        with patch("execution.close_all_positions") as mock_close:
            assert rm.check_absolute_drawdown(self._limit_equity - 100) is False
            mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# Daily stop: −0.75%, 2 trades, or 2 consecutive losses
# ---------------------------------------------------------------------------

class TestDailyStop:
    _limit_equity = BALANCE * (1 - config.DAILY_LOSS_PCT)   # 9 925.00

    def test_just_under_limit_allowed(self, rm):
        assert rm.check_daily_loss(self._limit_equity + 0.01) is True

    def test_exactly_at_limit_halts(self, rm):
        assert rm.check_daily_loss(self._limit_equity) is False
        assert rm.halted_today is True

    def test_over_limit_halts(self, rm):
        assert rm.check_daily_loss(self._limit_equity - 100) is False
        assert rm.halted_today is True

    def test_subsequent_calls_blocked_when_halted(self, rm):
        rm.halted_today = True
        assert rm.check_daily_loss(BALANCE + 500) is False

    def test_two_trades_per_day_blocks_third(self, rm):
        rm.record_trade_opened()
        rm.record_trade_opened()
        assert rm.trades_today == config.MAX_TRADES_PER_DAY
        ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is False
        assert msg == "MAX TRADES PER DAY"

    def test_two_consecutive_losses_halts_day(self, rm):
        rm.record_trade_result("LOSS")
        rm.record_trade_result("LOSS")
        ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is False
        assert msg == "DAILY CONSECUTIVE LOSSES"

    def test_win_resets_consecutive_losses(self, rm):
        rm.record_trade_result("LOSS")
        rm.record_trade_result("WIN")
        rm.record_trade_result("LOSS")
        assert rm.consec_losses_day == 1
        ok, _ = rm.can_trade(BALANCE, 0)
        assert ok is True

    def test_server_midnight_reset_clears_daily_state(self, rm):
        rm.halted_today = True
        rm.trades_today = 2
        rm.consec_losses_day = 2
        rm.daily_start_time = datetime.now(UTC) - timedelta(days=1)

        ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is True
        assert msg == "OK"
        assert rm.halted_today is False
        assert rm.trades_today == 0
        assert rm.consec_losses_day == 0

    def test_new_day_start_balance_anchors_to_current_equity(self, rm):
        rm.halted_today = True
        rm.daily_start_balance = BALANCE
        rm.daily_start_time = datetime.now(UTC) - timedelta(days=1)

        assert rm.check_daily_loss(9_910.0) is True
        assert rm.halted_today is False
        assert rm.daily_start_balance == 9_910.0


# ---------------------------------------------------------------------------
# Weekly stop: −1.5% or 5 consecutive losses, resets Monday (server time)
# ---------------------------------------------------------------------------

class TestWeeklyStop:
    _limit_equity = BALANCE * (1 - config.WEEKLY_STOP_PCT)   # 9 850.00

    def test_just_under_limit_allowed(self, rm):
        assert rm.check_weekly_loss(self._limit_equity + 0.01) is True

    def test_weekly_loss_halts(self, rm):
        assert rm.check_weekly_loss(self._limit_equity) is False
        assert rm.halted_this_week is True

    def test_five_consecutive_losses_halts_week(self, rm):
        for _ in range(config.MAX_CONSEC_LOSSES_WEEK):
            rm.record_trade_result("LOSS")
        # Daily gate would fire first in can_trade; check the weekly gate directly
        assert rm.check_weekly_loss(BALANCE) is False
        assert rm.halted_this_week is True

    def test_halt_persists_within_same_week(self, rm):
        rm.halted_this_week = True
        assert rm.check_weekly_loss(BALANCE + 500) is False

    def test_monday_reset_clears_weekly_state(self, rm):
        rm.halted_this_week = True
        rm.consec_losses_week = 5
        rm.week_start_time = datetime.now(UTC) - timedelta(days=8)
        rm.week_start_balance = BALANCE

        assert rm.check_weekly_loss(9_990.0) is True
        assert rm.halted_this_week is False
        assert rm.consec_losses_week == 0
        assert rm.week_start_balance == 9_990.0

    def test_weekly_gate_reason_in_can_trade(self, rm):
        equity = self._limit_equity - 1
        # Weekly loss (−1.5%) is worse than daily (−0.75%), so weekly is
        # checked first and must be the reported reason.
        ok, msg = rm.can_trade(equity, 0)
        assert ok is False
        assert msg == "WEEKLY STOP HIT"


# ---------------------------------------------------------------------------
# calculate_lot_size
# ---------------------------------------------------------------------------

class TestCalculateLotSize:
    def test_standard_calculation(self, rm):
        # lot = equity * RISK_PER_TRADE_PCT / (sl_pips * pip_value)
        expected = round(10_000 * config.RISK_PER_TRADE_PCT / (50 * 10.0), 2)
        lot = rm.calculate_lot_size(10_000, 1.0950, 1.1000, "EURUSD")
        assert lot == expected

    def test_lot_size_rounds_down_to_broker_step(self, rm):
        # 26.5-pip SL: raw lot = 30 / 265 ≈ 0.1132 → floors DOWN to 0.11
        lot = rm.calculate_lot_size(10_000, 1.09735, 1.1000, "EURUSD")
        raw = 10_000 * config.RISK_PER_TRADE_PCT / (26.5 * 10.0)
        assert lot == pytest.approx(0.11)
        assert lot <= raw  # never rounds up

    def test_entry_equals_sl_returns_zero(self, rm):
        assert rm.calculate_lot_size(10_000, 1.1000, 1.1000, "EURUSD") == 0.0

    def test_no_forced_minimum_lot_when_risk_size_too_small(self, tmp_path):
        rm = RiskManager(10, disabled_flag_path=tmp_path / "d.json")
        assert rm.calculate_lot_size(10, 1.0000, 1.5000, "EURUSD") == 0.0

    def test_maximum_clamp(self, rm):
        lot = rm.calculate_lot_size(10_000_000, 1.0999, 1.1000, "EURUSD")
        assert lot == 5.0

    def test_unsupported_symbol_returns_zero(self, rm):
        assert rm.calculate_lot_size(10_000, 1.0950, 1.1000, "NZDCAD") == 0.0

    def test_martingale_prevention(self, tmp_path):
        # Keep the equity decline within the warning threshold so this test
        # isolates the martingale cap, not the drawdown breaker.
        rm = RiskManager(9_600, disabled_flag_path=tmp_path / "d.json")
        rm._last_lot_size = 0.02
        rm._last_equity_at_lot = 9_600.0

        lot = rm.calculate_lot_size(9_500, 1.0950, 1.1000, "EURUSD")
        assert lot <= 0.02

    def test_no_martingale_cap_on_profit(self, rm):
        rm._last_lot_size = 0.05
        rm._last_equity_at_lot = 9_000.0

        lot = rm.calculate_lot_size(10_000, 1.0950, 1.1000, "EURUSD")
        expected = round(10_000 * config.RISK_PER_TRADE_PCT / (50 * 10.0), 2)
        assert lot == expected

    def test_heartbeat_risk_override(self, rm):
        rm.last_trade_time = datetime.now(UTC) - timedelta(days=config.HEARTBEAT_DAYS + 1)
        assert rm.heartbeat_active() is True
        assert rm.risk_pct_for_next_trade() == config.HEARTBEAT_RISK_PCT
        # 0.1% of 10 000 = $10 → 20-pip SL → 0.05 lots... 10/(20*10)=0.05
        lot = rm.calculate_lot_size(10_000, 1.0980, 1.1000, "EURUSD")
        assert lot == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Drawdown circuit breaker (soft reduction, must not mask the hard stop)
# ---------------------------------------------------------------------------

class TestDrawdownCircuitBreaker:
    _entry = 1.1000
    _sl = 1.0980   # 20 pips (inside the SL clamp)

    def test_full_size_below_warning(self, rm):
        equity = BALANCE * (1 - config.DRAWDOWN_WARNING_THRESHOLD + 0.001)
        assert rm._drawdown_size_multiplier(equity) == 1.0

    def test_half_size_at_warning(self, rm):
        equity = BALANCE * (1 - config.DRAWDOWN_WARNING_THRESHOLD)
        assert rm._drawdown_size_multiplier(equity) == 0.5

        full = rm.calculate_lot_size(BALANCE, self._sl, self._entry, "EURUSD")
        rm._last_lot_size = 0.0   # clear martingale state between calls
        reduced = rm.calculate_lot_size(equity, self._sl, self._entry, "EURUSD")
        assert reduced < full

    def test_sizing_blocked_at_kill_switch(self, rm):
        equity = BALANCE * (1 - config.KILL_SWITCH_PCT)
        assert rm._drawdown_size_multiplier(equity) == 0.0
        assert rm.calculate_lot_size(equity, self._sl, self._entry, "EURUSD") == 0.0

    def test_no_drawdown_in_profit(self, rm):
        assert rm.current_drawdown_pct(BALANCE + 500) == 0.0
        assert rm._drawdown_size_multiplier(BALANCE + 500) == 1.0


# ---------------------------------------------------------------------------
# calculate_sl_tp: SL 1.5×ATR beyond swing, clamp [8, 25] pips, TP = 2R
# ---------------------------------------------------------------------------

class TestCalculateSlTp:
    _entry = 1.10000

    def test_buy_sl_beyond_swing_tp_2r(self, rm):
        atr = 0.0008          # 8 pips → SL 12 pips below swing
        swing = 1.09950       # swing low 5 pips below entry → SL distance 17 pips
        result = rm.calculate_sl_tp(1, self._entry, atr, swing_price=swing)
        assert result is not None
        sl, tp = result
        assert sl == round(swing - config.SL_ATR_MULT * atr, 5)
        risk = self._entry - sl
        assert tp == round(self._entry + config.TP_R * risk, 5)
        assert sl < self._entry < tp

    def test_sell_sl_beyond_swing_tp_2r(self, rm):
        atr = 0.0008
        swing = 1.10050       # swing high 5 pips above entry
        result = rm.calculate_sl_tp(-1, self._entry, atr, swing_price=swing)
        assert result is not None
        sl, tp = result
        assert sl == round(swing + config.SL_ATR_MULT * atr, 5)
        risk = sl - self._entry
        assert tp == round(self._entry - config.TP_R * risk, 5)
        assert tp < self._entry < sl

    def test_sl_below_min_clamp_skips_trade(self, rm):
        # 1.5 × 4 pips = 6 pips < SL_MIN_PIPS (8) → skip
        assert rm.calculate_sl_tp(1, self._entry, 0.0004) is None

    def test_sl_above_max_clamp_skips_trade(self, rm):
        # 1.5 × 20 pips = 30 pips > SL_MAX_PIPS (25) → skip
        assert rm.calculate_sl_tp(1, self._entry, 0.0020) is None

    def test_zero_atr_skips_trade(self, rm):
        assert rm.calculate_sl_tp(1, self._entry, 0.0) is None

    def test_tp_is_r_multiple_of_actual_sl_distance_not_atr(self, rm):
        atr = 0.0008
        swing = 1.09920       # SL distance 8 + 12 = 20 pips; 2R = 40 pips
        sl, tp = rm.calculate_sl_tp(1, self._entry, atr, swing_price=swing)
        assert tp - self._entry == pytest.approx(2.0 * (self._entry - sl), abs=1e-9)
        assert tp - self._entry != pytest.approx(2.0 * atr, abs=1e-6)

    def test_breakeven_trigger_at_1r(self, rm):
        sl, _tp = rm.calculate_sl_tp(1, self._entry, 0.0008, swing_price=1.09950)
        be = rm.breakeven_trigger_price(1, self._entry, sl)
        assert be == round(self._entry + config.BE_AT_R * (self._entry - sl), 5)


# ---------------------------------------------------------------------------
# Heartbeat (21-day inactivity)
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_inactive_after_21_days(self, rm):
        rm.last_trade_time = datetime.now(UTC) - timedelta(days=config.HEARTBEAT_DAYS)
        assert rm.heartbeat_active() is True
        assert rm.adx_min_for_next_trade() == config.ADX_MIN_RELAXED

    def test_active_recent_trade(self, rm):
        rm.last_trade_time = datetime.now(UTC) - timedelta(days=1)
        assert rm.heartbeat_active() is False
        assert rm.risk_pct_for_next_trade() == config.RISK_PER_TRADE_PCT
        assert rm.adx_min_for_next_trade() == config.ADX_MIN

    def test_fresh_manager_uses_created_at(self, rm):
        assert rm.heartbeat_active() is False


# ---------------------------------------------------------------------------
# can_trade
# ---------------------------------------------------------------------------

class TestCanTrade:
    def test_all_clear(self, rm):
        ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is True
        assert msg == "OK"

    def test_daily_loss_breach(self, rm):
        # Loss worse than daily (−0.75%) but better than weekly (−1.5%)
        equity = BALANCE * (1 - config.DAILY_LOSS_PCT) - 1
        ok, msg = rm.can_trade(equity, 0)
        assert ok is False
        assert msg == "DAILY LOSS LIMIT HIT"

    def test_rollover_window(self, rm):
        with patch("risk_manager.is_rollover_window", return_value=True):
            ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is False
        assert msg == "ROLLOVER WINDOW"

    def test_server_no_trade_window(self, rm):
        with patch("risk_manager.is_no_trade_server_window", return_value=True):
            ok, msg = rm.can_trade(BALANCE, 0)
        assert ok is False
        assert msg == "ROLLOVER WINDOW"

    def test_max_trades_open(self, rm):
        ok, msg = rm.can_trade(BALANCE, config.MAX_OPEN_TRADES)
        assert ok is False
        assert msg == "MAX TRADES OPEN"

    def test_single_open_trade_blocks_new_entry(self, rm):
        """The5ers prohibits bulk trading: MAX_OPEN_TRADES must be 1."""
        assert config.MAX_OPEN_TRADES == 1
        ok, msg = rm.can_trade(BALANCE, 1)
        assert ok is False

    def test_priority_kill_switch_before_daily(self, rm):
        equity = BALANCE * (1 - config.KILL_SWITCH_PCT) - 1
        with patch("execution.close_all_positions"):
            ok, msg = rm.can_trade(equity, 0)
        assert msg == "KILL SWITCH HIT"


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

class TestIsRolloverWindow:
    def test_inside_window(self):
        fake_now = datetime(2023, 1, 1, config.ROLLOVER_START_UTC, 30, tzinfo=UTC)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            assert is_rollover_window() is True

    def test_outside_window_before(self):
        fake_now = datetime(2023, 1, 1, config.ROLLOVER_START_UTC - 1, 59, tzinfo=UTC)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            assert is_rollover_window() is False

    def test_outside_window_after(self):
        fake_now = datetime(2023, 1, 1, config.ROLLOVER_END_UTC, 0, tzinfo=UTC)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            assert is_rollover_window() is False


class TestNoTradeServerWindow:
    # config default window: 21:45 → 00:15 server time (spans midnight)

    def test_inside_before_midnight(self):
        assert is_no_trade_server_window(datetime(2026, 1, 5, 21, 45, tzinfo=UTC)) is True
        assert is_no_trade_server_window(datetime(2026, 1, 5, 23, 59, tzinfo=UTC)) is True

    def test_inside_after_midnight(self):
        assert is_no_trade_server_window(datetime(2026, 1, 6, 0, 14, tzinfo=UTC)) is True

    def test_outside(self):
        assert is_no_trade_server_window(datetime(2026, 1, 6, 0, 15, tzinfo=UTC)) is False
        assert is_no_trade_server_window(datetime(2026, 1, 5, 21, 44, tzinfo=UTC)) is False
        assert is_no_trade_server_window(datetime(2026, 1, 5, 12, 0, tzinfo=UTC)) is False
