"""
test_data_feed.py
"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from data_feed import (
    connect_mt5,
    get_ohlcv,
    get_latest_tick,
    get_account_info,
    save_data,
    load_data
)

@patch('data_feed.mt5')
@patch('data_feed.os.getenv')
def test_connect_mt5_success(mock_getenv, mock_mt5):
    # Mock environment variables
    mock_getenv.side_effect = lambda k: {
        "MT5_LOGIN": "123",
        "MT5_PASSWORD": "password",
        "MT5_SERVER": "Broker-Server"
    }.get(k)
    
    # Mock mt5 API correctly initialized
    mock_mt5.initialize.return_value = True
    
    # Mock mt5 account info returning object with balance
    mock_info = MagicMock()
    mock_info.balance = 5000.0
    mock_mt5.account_info.return_value = mock_info
    
    balance = connect_mt5()
    
    # Assert return logic and mt5.initialize arguments
    assert balance == 5000.0
    mock_mt5.initialize.assert_called_once_with(login=123, password="password", server="Broker-Server")

@patch('data_feed.mt5')
@patch('data_feed.os.getenv')
def test_connect_mt5_initialization_failure(mock_getenv, mock_mt5):
    # Simulate failed connection
    mock_getenv.side_effect = lambda k: {
        "MT5_LOGIN": "123",
        "MT5_PASSWORD": "password",
        "MT5_SERVER": "Broker-Server"
    }.get(k)
    
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (1, "Test error")
    
    with pytest.raises(ConnectionError, match="MT5 initialization failed"):
        connect_mt5()

@patch('data_feed.mt5')
def test_get_ohlcv_success(mock_mt5):
    mock_rates = [
        (1600000000, 1.1, 1.2, 1.0, 1.15, 100, 10, 0),
        (1600003600, 1.15, 1.25, 1.1, 1.2, 0, 20, 0),    # zero volume row, will be dropped
        (1600007200, 1.2, 1.3, 1.15, 1.25, 200, 15, 0)
    ]
    
    # Mocking MT5's structured array dtype
    dtype = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')]
    arr = np.array(mock_rates, dtype=dtype)
    mock_mt5.copy_rates_from_pos.return_value = arr
    
    # mt5.TIMEFRAME_H1 is part of mt5 module, we just trust the mapping works and data returns
    df = get_ohlcv("EURUSD", "H1", 3)
    
    assert len(df) == 2  # one row should be dropped because volume=0
    assert 'volume' in df.columns  # tick_volume was renamed
    assert 'tick_volume' not in df.columns
    assert str(df.index.tz) == 'UTC'
    assert df.loc[df.index[0], 'close'] == 1.15

@patch('data_feed.mt5')
def test_get_latest_tick_success(mock_mt5):
    mock_tick = MagicMock()
    mock_tick.ask = 1.10020
    mock_tick.bid = 1.10000
    mock_mt5.symbol_info_tick.return_value = mock_tick
    
    result = get_latest_tick("EURUSD")
    assert result['ask'] == 1.10020
    assert result['bid'] == 1.10000
    # Spread in pips (1.1002 - 1.1000) / 0.00001 = 20
    assert pytest.approx(result['spread']) == 20.0

@patch('data_feed.mt5')
def test_get_account_info_success(mock_mt5):
    mock_info = MagicMock()
    mock_info.balance = 10000.0
    mock_info.equity = 9500.0
    mock_info.margin = 500.0
    mock_info.margin_free = 9000.0
    mock_mt5.account_info.return_value = mock_info
    
    info = get_account_info()
    assert info['balance'] == 10000.0
    assert info['equity'] == 9500.0
    # Drawdown % calculates as (balance - equity)/balance * 100
    assert info['drawdown_pct'] == 5.0

@patch('pandas.DataFrame.to_csv')
@patch('os.makedirs')
def test_save_data(mock_makedirs, mock_to_csv):
    df = pd.DataFrame({"close": [1, 2, 3]})
    save_data(df, "test_file")
    
    mock_makedirs.assert_called_with("data", exist_ok=True)
    expected_path = os.path.join("data", "test_file.csv")
    mock_to_csv.assert_called_once_with(expected_path)

@patch('pandas.read_csv')
def test_load_data(mock_read_csv):
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    mock_read_csv.return_value = df
    
    loaded_df = load_data("test_file")
    expected_path = os.path.join("data", "test_file.csv")
    mock_read_csv.assert_called_once_with(expected_path, index_col='time', parse_dates=True)
    assert len(loaded_df) == 3
    assert str(loaded_df.index.tz) == 'UTC'
