"""
test_backtest_report.py

Unit tests for backtest_report.py functions.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from backtest_report import run_backtest, _sharpe_ratio, _profit_factor


# ── helper function tests ──
def test_sharpe_ratio_zero_std():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert _sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive():
    returns = pd.Series([1.0, 2.0, 1.5, 3.0])
    assert _sharpe_ratio(returns) > 0


def test_profit_factor_no_losses():
    returns = pd.Series([1.0, 2.0, 3.0])
    assert _profit_factor(returns) == float("inf")


def test_profit_factor_mixed():
    returns = pd.Series([10.0, -5.0])
    assert _profit_factor(returns) == 2.0


# ── run_backtest ──
def test_run_backtest_basic():
    """Creates deterministic data where BUY signals always hit TP."""
    np.random.seed(42)
    rows = 100

    # Steadily rising prices → model predicts BUY → TP always hit
    closes = np.linspace(1.1000, 1.1500, rows)
    highs = closes + 0.0020
    lows = closes - 0.0005
    atr = np.full(rows, 0.0010)

    df = pd.DataFrame({
        "close": closes,
        "high": highs,
        "low": lows,
        "atr_14": atr,
        "f1": np.random.randn(rows),
        "f2": np.random.randn(rows),
    })

    # Mock model that always predicts BUY with high confidence
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])

    # LabelEncoder: classes_ = [-1, 0, 1], so argmax=1 → inverse_transform → 0
    # We need argmax to map to signal=1 (buy). Set classes so index 1 = 1.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit([-1, 0, 1])  # classes_ = [-1, 0, 1]
    # argmax of [0.1, 0.8, 0.1] = index 1 → le.inverse_transform([1]) = 0 → hold
    # We need it to return BUY (1), so adjust proba:
    mock_model.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])
    # argmax = 2 → le.inverse_transform([2]) = 1 → BUY ✓

    with patch.object(config, "MIN_CONFIDENCE", 0.5):
        results = run_backtest(df, mock_model, le, starting_balance=10_000)

    assert results["total_trades"] > 0
    assert "win_rate" in results
    assert "total_pnl_pct" in results
    assert "max_drawdown_pct" in results
    assert "sharpe_ratio" in results
    assert "profit_factor" in results


def test_run_backtest_no_signal():
    """Model always returns hold → zero trades."""
    rows = 50
    df = pd.DataFrame({
        "close": np.ones(rows),
        "high": np.ones(rows) + 0.001,
        "low": np.ones(rows) - 0.001,
        "atr_14": np.full(rows, 0.001),
        "f1": np.zeros(rows),
    })

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit([-1, 0, 1])
    # argmax=0 → inverse_transform([0]) = -1 (sell), but confidence 0.8 > MIN. 
    # Actually we want *no* signal, so set confidence low:
    mock_model.predict_proba.return_value = np.array([[0.4, 0.3, 0.3]])

    with patch.object(config, "MIN_CONFIDENCE", 0.65):
        results = run_backtest(df, mock_model, le)

    assert results["total_trades"] == 0
    assert results["total_pnl_pct"] == 0.0
