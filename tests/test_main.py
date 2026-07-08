"""
test_main.py

Unit tests for main.py — the inverted rules-first flow:
strategy candidate → filters → optional MetaVeto → risk → execution.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strategy
from main import (
    initialize_bot,
    process_candle,
    get_current_m15_time,
    BotState,
)


UTC = timezone.utc

IN_SESSION = datetime(2026, 1, 14, 10, 0, tzinfo=UTC)  # Wednesday 10:00 London


def _candidate(direction=1):
    return strategy.Candidate(
        direction=direction,
        signal_time=pd.Timestamp("2026-01-14 09:45:00+00:00"),
        swing_price=1.0995 if direction == 1 else 1.1010,
        atr=0.0008,           # 8 pips → SL ≈ 12–17 pips (inside clamp)
        atr_median=0.0007,
    )


def _fresh_state(tmp_path, equity=10_000.0, meta_veto=None):
    from risk_manager import RiskManager

    return BotState(
        starting_balance=equity,
        risk=RiskManager(equity, disabled_flag_path=tmp_path / "d.json"),
        meta_veto=meta_veto,
    )


def _mock_session(monkeypatch):
    monkeypatch.setattr("main.datetime", MagicMock(now=lambda tz=None: IN_SESSION))


# ---------------------------------------------------------------------------
# get_current_m15_time
# ---------------------------------------------------------------------------

def test_get_current_m15_time_rounds_down():
    fake_now = datetime(2024, 6, 15, 14, 37, 22, tzinfo=UTC)
    with patch("main.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        result = get_current_m15_time()
    assert result.minute == 30
    assert result.second == 0


# ---------------------------------------------------------------------------
# initialize_bot
# ---------------------------------------------------------------------------

@patch("main.data_feed.connect_broker")
def test_initialize_bot_rules_only(mock_connect, capsys, monkeypatch):
    import config

    monkeypatch.setattr(config, "USE_META_VETO", False)
    mock_connect.return_value = 10_000.0

    state = initialize_bot(dry_run=True)

    assert state.starting_balance == 10_000.0
    assert state.meta_veto is None       # no model load in rules-only mode
    assert state.is_running is True

    captured = capsys.readouterr()
    assert "BOOTCAMP RULES BOT" in captured.out
    assert "DRY-RUN" in captured.out
    assert "MetaVeto: OFF" in captured.out


@patch("main.data_feed.connect_broker")
def test_initialize_bot_survives_missing_meta_model(mock_connect, monkeypatch):
    """USE_META_VETO with no model artifacts must degrade to rules-only."""
    import config

    monkeypatch.setattr(config, "USE_META_VETO", True)
    mock_connect.return_value = 10_000.0

    with patch("model.MetaVeto.load", side_effect=FileNotFoundError("no model")):
        state = initialize_bot(dry_run=True)

    assert state.meta_veto is None


# ---------------------------------------------------------------------------
# process_candle — dry-run happy path
# ---------------------------------------------------------------------------

@patch("main.trade_journal")
@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_process_candle_dry_run_logs_candidate(
    mock_data_feed, mock_news, mock_execution, mock_journal,
    caplog, tmp_path, monkeypatch,
):
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    mock_data_feed.get_latest_tick.return_value = {"ask": 1.1001, "bid": 1.1000}
    mock_data_feed.get_ohlcv.return_value = pd.DataFrame({"close": [1.1]})
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []
    mock_news.is_news_window.return_value = False

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None
    _mock_session(monkeypatch)
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: True)
    monkeypatch.setattr(
        "main.strategy.generate_candidate", lambda m15, h1, params=None: _candidate()
    )

    import logging
    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False), \
         caplog.at_level(logging.INFO, logger="main"):
        process_candle(state, dry_run=True)

    mock_execution.place_order.assert_not_called()
    assert any("[DRY RUN]" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# process_candle — gates and filters short-circuit
# ---------------------------------------------------------------------------

@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_blocked_by_news_filter(mock_data_feed, mock_news, mock_execution, tmp_path, monkeypatch):
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []
    mock_news.is_news_window.return_value = True

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: True)

    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        process_candle(state, dry_run=False)

    mock_data_feed.get_ohlcv.assert_not_called()   # never reaches the strategy
    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_blocked_by_kill_switch(mock_data_feed, mock_news, mock_execution, tmp_path):
    """−3% equity → kill switch fires before any strategy/news work."""
    mock_data_feed.get_account_info.return_value = {"equity": 9_690.0}
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None

    with patch("execution.close_all_positions"):
        process_candle(state, dry_run=False)

    assert state.risk.is_disabled()
    mock_news.is_news_window.assert_not_called()
    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_blocked_outside_session(mock_data_feed, mock_news, mock_execution, tmp_path, monkeypatch):
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: False)

    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        process_candle(state, dry_run=False)

    mock_news.is_news_window.assert_not_called()
    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_wide_spread_skips_candidate(mock_data_feed, mock_news, mock_execution, tmp_path, monkeypatch):
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    # 3-pip spread > MAX_SPREAD_PIPS (1.2)
    mock_data_feed.get_latest_tick.return_value = {"ask": 1.1003, "bid": 1.1000}
    mock_data_feed.get_ohlcv.return_value = pd.DataFrame({"close": [1.1]})
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []
    mock_news.is_news_window.return_value = False

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: True)
    monkeypatch.setattr(
        "main.strategy.generate_candidate", lambda m15, h1, params=None: _candidate()
    )

    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        process_candle(state, dry_run=False)

    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_meta_veto_blocks_candidate(mock_data_feed, mock_news, mock_execution, tmp_path, monkeypatch):
    """MetaVeto may block a rules candidate — and can never create one."""
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    mock_data_feed.get_latest_tick.return_value = {"ask": 1.1001, "bid": 1.1000}
    mock_data_feed.get_ohlcv.return_value = pd.DataFrame({"close": [1.1]})
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []
    mock_news.is_news_window.return_value = False

    veto = MagicMock()
    veto.allow.return_value = False
    state = _fresh_state(tmp_path, meta_veto=veto)
    state.risk.sync_counters_from_journal = lambda: None
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: True)
    monkeypatch.setattr(
        "main.strategy.generate_candidate", lambda m15, h1, params=None: _candidate()
    )
    monkeypatch.setattr(
        "features.get_live_features", lambda symbol: pd.Series({"f": 1.0})
    )

    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        process_candle(state, dry_run=False)

    veto.allow.assert_called_once()
    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_no_candidate_no_trade(mock_data_feed, mock_news, mock_execution, tmp_path, monkeypatch):
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    mock_data_feed.get_ohlcv.return_value = pd.DataFrame({"close": [1.1]})
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []
    mock_news.is_news_window.return_value = False

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: True)
    monkeypatch.setattr(
        "main.strategy.generate_candidate", lambda m15, h1, params=None: None
    )

    with patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        process_candle(state, dry_run=False)

    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.data_feed")
def test_live_order_placed_and_journalled(mock_data_feed, mock_news, mock_execution, tmp_path, monkeypatch):
    mock_data_feed.get_account_info.return_value = {"equity": 10_000.0}
    mock_data_feed.get_latest_tick.return_value = {"ask": 1.1001, "bid": 1.1000}
    mock_data_feed.get_ohlcv.return_value = pd.DataFrame({"close": [1.1]})
    mock_execution.count_open_trades.return_value = 0
    mock_execution.get_open_positions.return_value = []
    mock_execution.place_order.return_value = {"ticket": 42, "price": 1.1001}
    mock_news.is_news_window.return_value = False

    state = _fresh_state(tmp_path)
    state.risk.sync_counters_from_journal = lambda: None
    monkeypatch.setattr("main.strategy.entry_session_ok", lambda ts: True)
    monkeypatch.setattr(
        "main.strategy.generate_candidate", lambda m15, h1, params=None: _candidate()
    )

    with patch("main.trade_journal.log_trade") as mock_log, \
         patch("risk_manager.is_rollover_window", return_value=False), \
         patch("risk_manager.is_no_trade_server_window", return_value=False):
        process_candle(state, dry_run=False)

    mock_execution.place_order.assert_called_once()
    args = mock_execution.place_order.call_args[0]
    assert args[1] == 1                       # direction
    assert args[3] > 0 and args[4] > 0        # SL and TP always present
    mock_log.assert_called_once()
    assert state.risk.trades_today == 1


# ---------------------------------------------------------------------------
# manage_positions — flatten before major news
# ---------------------------------------------------------------------------

@patch("main.execution")
@patch("main.news_filter")
def test_flattens_open_positions_before_major_news(mock_news, mock_execution, tmp_path):
    from main import manage_positions

    mock_execution.get_open_positions.return_value = [{"ticket": 1}]
    mock_news.should_flatten_for_news.return_value = True

    manage_positions(_fresh_state(tmp_path), dry_run=False)

    mock_execution.close_all_positions.assert_called_once()
    mock_execution.manage_breakeven.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
def test_breakeven_managed_when_no_news(mock_news, mock_execution, tmp_path):
    from main import manage_positions

    mock_execution.get_open_positions.return_value = [{"ticket": 1}]
    mock_news.should_flatten_for_news.return_value = False

    manage_positions(_fresh_state(tmp_path), dry_run=False)

    mock_execution.close_all_positions.assert_not_called()
    mock_execution.manage_breakeven.assert_called_once()
