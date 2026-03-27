"""
test_backtest.py

Unit tests for the backtest.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backtest import run_backtest
import config

@patch('backtest.connect_mt5')
@patch('backtest.get_historical_data')
@patch('backtest.disconnect_mt5')
def test_run_backtest_insufficient_data(mock_disconnect, mock_get_historical_data, mock_connect):
    mock_connect.return_value = True
    
    # Create small dummy data (not enough for features + lookforward)
    rows = 50
    df = pd.DataFrame({
        'open': np.random.randn(rows).cumsum() + 100,
        'high': np.random.randn(rows).cumsum() + 101,
        'low': np.random.randn(rows).cumsum() + 99,
        'close': np.random.randn(rows).cumsum() + 100,
        'tick_volume': np.random.randint(100, 1000, rows)
    })
    
    mock_get_historical_data.return_value = df
    
    # Run backtest, should return empty dict because not enough data
    results = run_backtest(symbol="EURUSD", timeframe=15, num_candles=rows)
    
    assert isinstance(results, dict)
    assert len(results) == 0

@patch('backtest.connect_mt5')
@patch('backtest.get_historical_data')
@patch('backtest.disconnect_mt5')
def test_run_backtest_failure(mock_disconnect, mock_get_historical_data, mock_connect):
    mock_connect.return_value = False
    
    # Connection failure
    results = run_backtest("EURUSD", 15, 1000)
    assert results == {}
