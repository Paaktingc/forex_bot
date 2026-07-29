"""
tests/test_deployment_safety.py

Regression tests for the live/challenge deployment blockers (Findings 1–7):

  1. Risk baseline survives a restart (never re-anchored to a lower balance).
  2. Broker-side SL/TP closes are reconciled into results.
  4. can_trade() halts when the step profit target is reached.
  5. Official daily-loss check is enforced in can_trade().
  6. The still-forming candle is excluded from OHLCV.
  7. process_candle() signals transient failure so the candle is retried.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from risk_manager import RiskManager
from reconcile import classify_closure, reconcile_closures
from bar_utils import drop_forming_bar

UTC = timezone.utc


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Point the persisted risk state + disabled flag at a temp dir."""
    state = tmp_path / "risk_state.json"
    disabled = tmp_path / "disabled.json"
    monkeypatch.setattr(config, "RISK_STATE_PATH", str(state), raising=False)
    return {"state": state, "disabled": disabled}


# ---------------------------------------------------------------------------
# Finding 1 — persistent baseline across restart
# ---------------------------------------------------------------------------

def test_baseline_survives_restart_after_loss(tmp_state):
    rm1 = RiskManager.load_or_init(5000.0, disabled_flag_path=tmp_state["disabled"])
    assert rm1.starting_balance == 5000.0
    assert tmp_state["state"].exists()

    # Restart with a LOWER broker balance (account drew down while offline).
    rm2 = RiskManager.load_or_init(4900.0, disabled_flag_path=tmp_state["disabled"])
    # Baseline must remain the persisted step-start, NOT the new lower balance.
    assert rm2.starting_balance == 5000.0
    # Drawdown is measured from the true baseline (2%), not reset to 0.
    assert rm2.current_drawdown_pct(4900.0) == pytest.approx(0.02, abs=1e-6)


def test_new_step_flag_rebaselines(tmp_state):
    RiskManager.load_or_init(5000.0, disabled_flag_path=tmp_state["disabled"])
    rm = RiskManager.load_or_init(
        10000.0, reset=True, disabled_flag_path=tmp_state["disabled"]
    )
    assert rm.starting_balance == 10000.0
    saved = json.loads(tmp_state["state"].read_text())
    assert saved["starting_balance"] == 10000.0


def test_programme_mismatch_starts_fresh(tmp_state, monkeypatch):
    # Persist a state file tagged for a different programme.
    tmp_state["state"].write_text(json.dumps({
        "version": 1, "programme": "high_stakes", "starting_balance": 999.0,
        "created_at": datetime.now(UTC).isoformat(),
        "daily_start_balance": 999.0, "daily_start_time": datetime.now(UTC).isoformat(),
        "week_start_balance": 999.0, "week_start_time": datetime.now(UTC).isoformat(),
    }))
    monkeypatch.setattr(config, "PROGRAMME", "bootcamp", raising=False)
    rm = RiskManager.load_or_init(5000.0, disabled_flag_path=tmp_state["disabled"])
    # Must NOT adopt the mismatched programme's baseline.
    assert rm.starting_balance == 5000.0


# ---------------------------------------------------------------------------
# Finding 2 — broker-close reconciliation
# ---------------------------------------------------------------------------

def _row(**kw):
    base = {"ticket": 1, "direction": "BUY", "entry_price": 1.1000,
            "sl": 1.0950, "tp": 1.1100, "lot_size": 0.10}
    base.update(kw)
    return base


def test_classify_win_uses_tp():
    price, reason, result = classify_closure(_row(), last_profit=12.0)
    assert reason == "TP" and result == "WIN" and price == 1.1100


def test_classify_loss_uses_sl():
    price, reason, result = classify_closure(_row(), last_profit=-8.0)
    assert reason == "SL" and result == "LOSS" and price == 1.0950


def test_classify_unknown_assumes_sl_loss():
    price, reason, result = classify_closure(_row(), last_profit=None)
    assert result == "LOSS" and reason == "RECONCILED_ASSUMED_SL" and price == 1.0950


def test_classify_breakeven_when_sl_at_entry():
    # SL trailed to entry (breakeven), closed flat/slightly negative.
    price, reason, result = classify_closure(
        _row(sl=1.1000), last_profit=-0.2
    )
    assert result == "BE" and reason == "BREAKEVEN"


def test_reconcile_feeds_results_into_risk(tmp_state):
    rm = RiskManager.load_or_init(5000.0, disabled_flag_path=tmp_state["disabled"])
    exits = []
    journal_open = [_row(ticket=1), _row(ticket=2, direction="SELL")]
    # Ticket 1 closed at broker (not in open set); ticket 2 still open.
    recs = reconcile_closures(
        journal_open=journal_open,
        broker_open_tickets={2},
        last_profit_by_ticket={1: -5.0},
        on_exit=lambda t, p, r: exits.append((t, p, r)),
        on_result=rm.record_trade_result,
    )
    assert len(recs) == 1 and recs[0]["ticket"] == 1
    assert exits and exits[0][0] == 1
    # The loss propagated into the consecutive-loss counter.
    assert rm.consec_losses_day == 1


# ---------------------------------------------------------------------------
# Finding 4 — profit-target stop
# ---------------------------------------------------------------------------

def test_can_trade_halts_at_profit_target(tmp_state, monkeypatch):
    monkeypatch.setattr(config, "PROFIT_TARGET_PCT", 0.06, raising=False)
    monkeypatch.setattr(config, "MIN_PROFITABLE_DAYS", 0, raising=False)
    rm = RiskManager.load_or_init(5000.0, disabled_flag_path=tmp_state["disabled"])
    ok, reason = rm.can_trade(5000.0 * 1.061, open_trades=0)
    assert ok is False and "TARGET" in reason
    assert rm.is_disabled() is True  # passed-step flag persisted
    tmp_state["disabled"].unlink(missing_ok=True)


def test_profit_target_waits_for_profitable_days(tmp_state, monkeypatch):
    monkeypatch.setattr(config, "PROFIT_TARGET_PCT", 0.10, raising=False)
    monkeypatch.setattr(config, "MIN_PROFITABLE_DAYS", 3, raising=False)
    rm = RiskManager.load_or_init(5000.0, disabled_flag_path=tmp_state["disabled"])
    # Target hit on equity but profitable-day requirement not met yet.
    assert rm.profit_target_reached(5000.0 * 1.11) is False
    rm.profitable_days = 3
    assert rm.profit_target_reached(5000.0 * 1.11) is True


# ---------------------------------------------------------------------------
# Finding 5 — official daily-loss enforced in can_trade
# ---------------------------------------------------------------------------

def test_official_daily_loss_blocks_in_can_trade(tmp_state, monkeypatch):
    monkeypatch.setattr(config, "OFFICIAL_DAILY_LOSS_PCT", 0.05, raising=False)
    monkeypatch.setattr(config, "PROFIT_TARGET_PCT", 0.10, raising=False)
    rm = RiskManager.load_or_init(100000.0, disabled_flag_path=tmp_state["disabled"])
    rm.daily_start_balance = 100000.0
    ok, reason = rm.can_trade(94000.0, open_trades=0)  # 6% down on the day
    assert ok is False
    tmp_state["disabled"].unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Finding 6 — forming candle excluded
# ---------------------------------------------------------------------------

def _bars(start: datetime, n: int, minutes: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [start + timedelta(minutes=minutes * i) for i in range(n)], tz="UTC"
    )
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": 1.0}, index=idx)


def test_drop_forming_bar_removes_unclosed():
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    df = _bars(start, 3, 15)                       # 09:00, 09:15, 09:30
    now = start + timedelta(minutes=35)            # 09:35 → 09:30 bar still forming
    out = drop_forming_bar(df, 15, now=now)
    assert len(out) == 2
    assert out.index[-1] == start + timedelta(minutes=15)  # last CLOSED bar


def test_drop_forming_bar_noop_when_all_closed():
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    df = _bars(start, 3, 15)
    now = start + timedelta(minutes=60)            # everything closed
    out = drop_forming_bar(df, 15, now=now)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Finding 7 — process_candle returns a retry signal
# ---------------------------------------------------------------------------

def test_process_candle_returns_false_on_broker_failure(monkeypatch):
    import main

    # Empty account info = transient broker failure → retryable.
    monkeypatch.setattr(main.data_feed, "get_account_info", lambda: {})
    state = object.__new__(main.BotState)  # no broker needed for this path
    assert main.process_candle(state, dry_run=True) is False
