"""
test_trade_journal.py

Unit tests for the trade_journal.py module.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import trade_journal


FIXED_NOW = datetime(2026, 3, 27, 12, 0, tzinfo=trade_journal.UTC)


@pytest.fixture(autouse=True)
def journal_file(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_journal, "JOURNAL_FILE", tmp_path / "test_trades.csv")
    monkeypatch.setattr(trade_journal, "_timestamp_now", lambda: FIXED_NOW)
    monkeypatch.setattr(trade_journal.config, "STARTING_BALANCE", 10_000.0)


def _entry(
    ticket: int,
    *,
    timestamp: datetime | None = None,
    symbol: str = "EURUSD",
    direction: str = "BUY",
    entry_price: float = 1.1000,
    sl: float = 1.0950,
    tp: float = 1.1050,
    lot_size: float = 0.1,
    confidence: float = 0.8,
    drawdown_at_entry: float = 0.5,
    session: str = "LONDON",
) -> dict:
    return {
        "ticket": ticket,
        "timestamp": (timestamp or FIXED_NOW).isoformat(),
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "sl": sl,
        "tp": tp,
        "lot_size": lot_size,
        "confidence": confidence,
        "drawdown_at_entry": drawdown_at_entry,
        "session": session,
    }


def test_log_entry_creates_requested_schema():
    trade_journal.log_entry(_entry(123))

    df = pd.read_csv(trade_journal.JOURNAL_FILE)

    assert list(df.columns) == trade_journal.JOURNAL_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "ticket"] == 123
    assert df.loc[0, "direction"] == "BUY"
    assert df.loc[0, "lot_size"] == 0.1
    assert df.loc[0, "confidence"] == 0.8
    assert pd.isna(df.loc[0, "exit_price"])
    assert pd.isna(df.loc[0, "result"])


def test_log_exit_updates_trade_metrics():
    trade_journal.log_entry(_entry(123))
    trade_journal.log_exit(123, 1.1050, "tp")

    df = pd.read_csv(trade_journal.JOURNAL_FILE)

    assert df.loc[0, "exit_price"] == 1.1050
    assert df.loc[0, "pnl_pips"] == 50.0
    assert df.loc[0, "pnl_currency"] == 50.0
    assert df.loc[0, "result"] == "WIN"
    assert df.loc[0, "exit_reason"] == "TP"
    assert pd.notna(df.loc[0, "exit_time"])


def test_get_daily_summary_uses_today_entries_only():
    trade_journal.log_entry(_entry(1, direction="BUY", session="LONDON"))
    trade_journal.log_exit(1, 1.1050, "TP")

    trade_journal.log_entry(
        _entry(
            2,
            direction="SELL",
            entry_price=1.2000,
            sl=1.2050,
            tp=1.1900,
            session="NEW_YORK",
        )
    )
    trade_journal.log_exit(2, 1.2025, "MANUAL")

    trade_journal.log_entry(_entry(3, timestamp=FIXED_NOW - timedelta(days=1), session="ASIA"))
    trade_journal.log_exit(3, 1.1050, "TP")

    trade_journal.log_entry(
        _entry(
            4,
            direction="BUY",
            entry_price=1.1500,
            sl=1.1450,
            tp=1.1600,
            session="OVERLAP",
        )
    )

    summary = trade_journal.get_daily_summary()

    assert summary == {
        "total_trades": 3,
        "wins": 1,
        "losses": 1,
        "win_rate": 50.0,
        "total_pnl": 25.0,
        "max_single_loss": -25.0,
        "avg_rr": 0.25,
    }


def test_get_overall_stats_aggregates_full_journal():
    trade_journal.log_entry(_entry(1, confidence=0.8, session="LONDON"))
    trade_journal.log_exit(1, 1.1050, "TP")

    trade_journal.log_entry(
        _entry(
            2,
            direction="SELL",
            entry_price=1.2000,
            sl=1.2050,
            tp=1.1900,
            confidence=0.7,
            session="NEW_YORK",
        )
    )
    trade_journal.log_exit(2, 1.2025, "MANUAL")

    trade_journal.log_entry(
        _entry(
            3,
            entry_price=1.3000,
            sl=1.2950,
            tp=1.3050,
            lot_size=0.2,
            confidence=0.9,
            session="LONDON",
        )
    )
    trade_journal.log_exit(3, 1.3050, "TP")

    stats = trade_journal.get_overall_stats()

    assert stats["total_trades"] == 3
    assert stats["win_rate"] == pytest.approx(66.67, abs=0.01)
    assert stats["total_pnl_pct"] == 1.25
    assert stats["max_drawdown"] == 25.0
    assert stats["profit_factor"] == 6.0
    assert stats["avg_confidence"] == 0.8
    assert stats["best_session"] == "LONDON"
    assert stats["sharpe_ratio"] > 0


def test_print_dashboard_renders_terminal_summary(monkeypatch, capsys):
    trade_journal.log_entry(_entry(123, session="LONDON"))
    trade_journal.log_exit(123, 1.1050, "TP")

    monkeypatch.setattr(
        trade_journal,
        "_get_account_snapshot",
        lambda: {"balance": 10_050.0, "equity": 10_025.0, "drawdown_pct": 0.25},
    )
    monkeypatch.setattr(trade_journal, "_get_open_trade_count", lambda: 1)

    trade_journal.print_dashboard()
    output = capsys.readouterr().out

    assert "THE5ERS ML BOT DASHBOARD" in output
    assert "10,050.00" in output
    assert "10,025.00" in output
    assert "0.25%" in output
    assert "+50.00" in output
    assert "100.0%" in output
    assert "Open Trades:" in output


def test_legacy_log_trade_and_log_closing_still_work():
    trade_journal.log_trade(99, "EURUSD", "BUY", 0.1, 1.1000, 1.0950, 1.1050)
    trade_journal.log_closing(99, 1.1020, 20.0)

    df = pd.read_csv(trade_journal.JOURNAL_FILE)

    assert len(df) == 1
    assert df.loc[0, "ticket"] == 99
    assert df.loc[0, "pnl_currency"] == 20.0
    assert df.loc[0, "result"] == "WIN"
    assert df.loc[0, "exit_reason"] == "MANUAL"
