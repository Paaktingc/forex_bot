"""
test_main.py

Unit tests for main.py. Tests the orchestration logic using mocks.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (
    initialize_bot,
    process_candle,
    get_current_m15_time,
    BotState,
)


UTC = timezone.utc


# ---------------------------------------------------------------------------
# get_current_m15_time
# ---------------------------------------------------------------------------

def test_get_current_m15_time_rounds_down():
    """M15 boundary should round down to nearest 15 min."""
    fake_now = datetime(2024, 6, 15, 14, 37, 22, tzinfo=UTC)
    with patch("main.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        # .replace should be forwarded to the real datetime method
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        result = get_current_m15_time()
    assert result.minute == 30
    assert result.second == 0


# ---------------------------------------------------------------------------
# initialize_bot
# ---------------------------------------------------------------------------

@patch("main.model_module.load_model")
@patch("main.data_feed.connect_mt5")
def test_initialize_bot(mock_connect, mock_load, capsys):
    mock_connect.return_value = 10_000.0
    mock_model = MagicMock()
    mock_le = MagicMock()
    mock_load.return_value = (mock_model, mock_le)

    state = initialize_bot(dry_run=True)

    assert state.starting_balance == 10_000.0
    assert state.model is mock_model
    assert state.label_encoder is mock_le
    assert state.is_running is True

    # Startup banner should print
    captured = capsys.readouterr()
    assert "THE5ERS ML FOREX BOT" in captured.out
    assert "DRY-RUN" in captured.out


# ---------------------------------------------------------------------------
# process_candle — full dry-run path
# ---------------------------------------------------------------------------

@patch("main.trade_journal")
@patch("main.execution")
@patch("main.features")
@patch("main.news_filter")
@patch("main.data_feed")
@patch("main.mt5")
def test_process_candle_dry_run_logs_signal(
    mock_mt5, mock_data_feed, mock_news, mock_features,
    mock_execution, mock_journal, caplog,
):
    """When all checks pass and model returns a signal, dry-run should log but not place order."""
    # Account equity
    acct = MagicMock()
    acct.equity = 10_000.0
    mock_mt5.account_info.return_value = acct

    # Execution: no open trades
    mock_execution.count_open_trades.return_value = 0

    # News filter: no news
    mock_news.is_news_window.return_value = False

    # Features
    feature_series = pd.Series(
        np.random.randn(21),
        index=[f"feat_{i}" for i in range(21)],
    )
    mock_features.get_live_features.return_value = feature_series

    # Model prediction
    mock_model = MagicMock()
    mock_le = MagicMock()

    from risk_manager import RiskManager
    risk = RiskManager(10_000.0)

    state = BotState(
        starting_balance=10_000.0,
        model=mock_model,
        label_encoder=mock_le,
        risk=risk,
    )

    # Mock model_module.predict_signal to return BUY
    with patch("main.model_module.predict_signal", return_value=(1, 0.85)):
        # Mock data_feed.get_ohlcv for ATR
        ohlcv_df = pd.DataFrame({
            "open": [1.1] * 20,
            "high": [1.11] * 20,
            "low": [1.09] * 20,
            "close": [1.10] * 20,
            "volume": [100] * 20,
        })
        mock_data_feed.get_ohlcv.return_value = ohlcv_df

        # Mock features.compute_indicators
        ind_df = ohlcv_df.copy()
        ind_df["atr_14"] = 0.001
        mock_features.compute_indicators.return_value = ind_df

        # Mock tick
        mock_data_feed.get_latest_tick.return_value = {"ask": 1.1001, "bid": 1.1000}

        # Rollover: not in window
        with patch("risk_manager.is_rollover_window", return_value=False):
            import logging
            with caplog.at_level(logging.INFO, logger="main"):
                process_candle(state, dry_run=True)

    # Should NOT place any real order
    mock_execution.place_order.assert_not_called()

    # Should log DRY RUN
    assert any("[DRY RUN]" in record.message for record in caplog.records)


@patch("main.execution")
@patch("main.news_filter")
@patch("main.mt5")
def test_process_candle_blocked_by_news_filter(
    mock_mt5, mock_news, mock_execution,
):
    """When news filter blocks, process_candle should return early."""
    acct = MagicMock()
    acct.equity = 10_000.0
    mock_mt5.account_info.return_value = acct
    mock_execution.count_open_trades.return_value = 0

    # News filter blocks trading
    mock_news.is_news_window.return_value = True

    from risk_manager import RiskManager
    state = BotState(
        starting_balance=10_000.0,
        model=MagicMock(),
        label_encoder=MagicMock(),
        risk=RiskManager(10_000.0),
    )

    with patch("risk_manager.is_rollover_window", return_value=False):
        process_candle(state, dry_run=False)

    # Should never reach order placement
    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.mt5")
def test_process_candle_blocked_by_drawdown(
    mock_mt5, mock_news, mock_execution,
):
    """When drawdown limit is breached, process_candle should return early."""
    acct = MagicMock()
    acct.equity = 9_000.0  # well below 4.5% drawdown
    mock_mt5.account_info.return_value = acct
    mock_execution.count_open_trades.return_value = 0

    from risk_manager import RiskManager
    state = BotState(
        starting_balance=10_000.0,
        model=MagicMock(),
        label_encoder=MagicMock(),
        risk=RiskManager(10_000.0),
    )

    with patch("execution.close_all_positions"):
        process_candle(state, dry_run=False)

    # Should never call news filter or place order
    mock_news.is_news_window.assert_not_called()
    mock_execution.place_order.assert_not_called()


@patch("main.execution")
@patch("main.news_filter")
@patch("main.mt5")
def test_process_candle_rollover_blocks_trading(
    mock_mt5, mock_news, mock_execution,
):
    """When rollover window is active, can_trade() returns False."""
    acct = MagicMock()
    acct.equity = 10_000.0
    mock_mt5.account_info.return_value = acct
    mock_execution.count_open_trades.return_value = 0

    from risk_manager import RiskManager
    state = BotState(
        starting_balance=10_000.0,
        model=MagicMock(),
        label_encoder=MagicMock(),
        risk=RiskManager(10_000.0),
    )

    with patch("risk_manager.is_rollover_window", return_value=True):
        process_candle(state, dry_run=False)

    mock_news.is_news_window.assert_not_called()
    mock_execution.place_order.assert_not_called()
