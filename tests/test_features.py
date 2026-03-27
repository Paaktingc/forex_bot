"""
test_features.py

Unit tests for the features.py module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from features import add_technical_indicators

def create_dummy_data(rows=250):
    """Creates dummy OHLCV data for testing indicators."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=rows, freq='15min')
    df = pd.DataFrame({
        'open': np.random.randn(rows).cumsum() + 100,
        'high': np.random.randn(rows).cumsum() + 101,
        'low': np.random.randn(rows).cumsum() + 99,
        'close': np.random.randn(rows).cumsum() + 100,
        'tick_volume': np.random.randint(100, 1000, rows)
    }, index=dates)
    return df

def test_add_technical_indicators_success():
    df = create_dummy_data(300)
    df_features = add_technical_indicators(df)
    
    assert not df_features.empty
    
    # Check if indicator columns exist
    cols_str = " ".join(df_features.columns)
    assert 'EMA' in cols_str
    assert 'RSI' in cols_str
    assert 'MACD' in cols_str
    assert 'ATR' in cols_str
    
    # Should have fewer rows due to dropna from rolling windows (e.g. SMA 200 drops 199 rows)
    assert len(df_features) < len(df)
    assert df_features.isna().sum().sum() == 0

def test_add_technical_indicators_small_df():
    df = create_dummy_data(10)
    df_features = add_technical_indicators(df)
    
    # Should return original if too small
    assert len(df_features) == 10
    cols_str = " ".join(df_features.columns)
    assert 'RSI' not in cols_str
