"""
test_integration_killswitch.py

Integration tests for the kill-switch chain:
  synthetic 12-loss streak ⇒ RiskManager flattens and disables at −3%,
  never letting equity gating reach the official −5%; a restart while
  disabled stays disabled until the flag file is deleted manually.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from risk_manager import RiskManager

UTC = timezone.utc
BALANCE = 10_000.0
LOSS_PER_TRADE = 0.0033  # −0.33% per losing trade (0.3% risk + costs)


@pytest.fixture(autouse=True)
def _quiet_windows():
    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        yield


def test_twelve_loss_streak_disables_at_minus_3_never_minus_5(tmp_path):
    flag = tmp_path / "disabled.json"
    rm = RiskManager(BALANCE, disabled_flag_path=flag)

    equity = BALANCE
    disabled_at_equity = None
    flatten_calls = []

    with patch("execution.close_all_positions", side_effect=lambda: flatten_calls.append(equity)):
        for trade_number in range(1, 13):
            # Fresh day/week so daily-pacing halts don't hide the kill switch
            # (worst case: pacing limits fail; the kill switch must still fire)
            rm.daily_start_time = datetime.now(UTC) - timedelta(days=1)
            rm.week_start_time = datetime.now(UTC)
            rm.halted_today = False
            rm.trades_today = 0
            rm.consec_losses_day = 0

            ok, reason = rm.can_trade(equity, 0)
            if rm.is_disabled():
                disabled_at_equity = equity
                assert reason == "KILL SWITCH HIT"
                break
            if ok:
                rm.record_trade_opened()
            equity *= 1.0 - LOSS_PER_TRADE
            rm.record_trade_result("LOSS")

    # The kill switch fired...
    assert rm.is_disabled(), "kill switch never fired during a 12-loss streak"
    assert disabled_at_equity is not None
    dd_at_disable = (BALANCE - disabled_at_equity) / BALANCE
    # ... at −3% (within one trade's worth of slack), far above the −5% limit
    assert config.KILL_SWITCH_PCT <= dd_at_disable < config.KILL_SWITCH_PCT + LOSS_PER_TRADE
    assert dd_at_disable < config.MAX_DRAWDOWN_LIMIT

    # ... and flattened all open positions exactly once
    assert len(flatten_calls) == 1

    # Once disabled, no equity level re-enables trading
    ok, reason = rm.can_trade(BALANCE, 0)
    assert ok is False
    assert "DISABLED" in reason


def test_restart_while_disabled_stays_disabled(tmp_path):
    flag = tmp_path / "disabled.json"
    rm = RiskManager(BALANCE, disabled_flag_path=flag)
    with patch("execution.close_all_positions"):
        rm.check_kill_switch(BALANCE * (1 - config.KILL_SWITCH_PCT))
    assert rm.is_disabled()

    # Simulated restart: new process, new RiskManager, same flag file
    rm2 = RiskManager(BALANCE, disabled_flag_path=flag)
    assert rm2.is_disabled()
    ok, reason = rm2.can_trade(BALANCE, 0)
    assert ok is False
    assert "DISABLED" in reason

    # Manual re-arm: delete the flag file → trading allowed again
    flag.unlink()
    rm3 = RiskManager(BALANCE, disabled_flag_path=flag)
    ok, reason = rm3.can_trade(BALANCE, 0)
    assert ok is True


def test_journal_fed_counters_trigger_daily_halt(tmp_path, monkeypatch):
    """Two journalled losses today ⇒ counters sync ⇒ daily halt."""
    import trade_journal

    journal = tmp_path / "trades.csv"
    monkeypatch.setattr(trade_journal, "JOURNAL_FILE", str(journal))

    now = datetime.now(UTC)
    for i, ts in enumerate([now - timedelta(hours=2), now - timedelta(hours=1)]):
        trade_journal.log_entry(
            {
                "ticket": 100 + i,
                "timestamp": ts,
                "symbol": "EURUSD",
                "direction": "BUY",
                "entry_price": 1.1000,
                "sl": 1.0985,
                "tp": 1.1030,
                "lot_size": 0.05,
            }
        )
        trade_journal.log_exit(100 + i, 1.0985, "SL")

    rm = RiskManager(BALANCE, disabled_flag_path=tmp_path / "d.json")
    rm.sync_counters_from_journal()

    assert rm.trades_today == 2
    assert rm.consec_losses_day == 2

    ok, reason = rm.can_trade(BALANCE, 0)
    assert ok is False
    assert reason in ("MAX TRADES PER DAY", "DAILY CONSECUTIVE LOSSES")
