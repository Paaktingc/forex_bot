"""
test_data_feed.py

Unit tests for the data_feed.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_feed import connect_mt5, disconnect_mt5, get_historical_data, get_current_tick, get_account_info

@patch('data_feed.mt5')
@patch('data_feed.os.getenv')
def test_connect_mt5_success(mock_getenv, mock_mt5):
    mock_getenv.side_effect = lambda key, default=None: {
        "MT5_PATH": "path",
        "MT5_SERVER": "server",
        "MT5_ACCOUNT": "123",
        "MT5_PASSWORD": "pass"
    }.get(key, default)
    
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True
    
    assert connect_mt5() == True
    mock_mt5.initialize.assert_called_once_with(path="path")
    mock_mt5.login.assert_called_once_with(login=123, password="pass", server="server")

@patch('data_feed.mt5')
def test_get_historical_data_success(mock_mt5):
    mock_rates = [
        (1600000000, 1.1, 1.2, 1.0, 1.15, 100, 10, 0),
        (1600003600, 1.15, 1.25, 1.1, 1.2, 200, 20, 0)
    ]
    # Create structured array like MT5 returns
    dtype = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')]
    arr = np.array(mock_rates, dtype=dtype)
    
    mock_mt5.copy_rates_from_pos.return_value = arr
    
    df = get_historical_data("EURUSD", 1, 2)
    assert df is not None
    assert len(df) == 2
    assert 'close' in df.columns

@patch('data_feed.mt5')
def test_get_historical_data_failure(mock_mt5):
    mock_mt5.copy_rates_from_pos.return_value = None
    mock_mt5.last_error.return_value = (1, "Error")
    
    df = get_historical_data("EURUSD", 1, 2)
    assert df is None

@patch('data_feed.mt5')
def test_get_current_tick_success(mock_mt5):
    mock_tick = MagicMock()
    mock_tick.bid = 1.1000
    mock_tick.ask = 1.1002
    mock_tick.time = 1600000000
    mock_mt5.symbol_info_tick.return_value = mock_tick
    
    tick = get_current_tick("EURUSD")
    assert tick is not None
    assert tick['bid'] == 1.1000
    assert tick['ask'] == 1.1002

@patch('data_feed.mt5')
def test_get_account_info_success(mock_mt5):
    mock_info = MagicMock()
    mock_info._asdict.return_value = {'equity': 10000, 'balance': 10000}
    mock_mt5.account_info.return_value = mock_info
    
    info = get_account_info()
    assert info is not None
    assert info['equity'] == 10000
