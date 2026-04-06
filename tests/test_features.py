import sys
import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch

# Mock MetaTrader5 before importing features if it's not available in the testing environment
# But the project already has an env for it, let's assume it imports successfully or we can patch if needed.
# Let's just import normally first.
from features import (
    compute_indicators,
    compute_session_features,
    add_h1_trend,
    build_feature_matrix,
    get_live_features,
    FEATURE_COLS
)

def generate_synthetic_data(num_bars: int, freq: str = "15min") -> pd.DataFrame:
    """Generates synthetic OHLCV data for testing."""
    np.random.seed(42)
    base_time = pd.Timestamp("2023-01-01 00:00:00")
    # Using '15min' or '1h' works in pandas
    times = [base_time + pd.Timedelta(freq) * i for i in range(num_bars)]
    
    returns = np.random.normal(0, 0.001, num_bars)
    close_prices = 1.1000 * np.exp(np.cumsum(returns))
    
    high_prices = close_prices + np.random.uniform(0.0001, 0.0020, num_bars)
    low_prices = close_prices - np.random.uniform(0.0001, 0.0020, num_bars)
    open_prices = close_prices - np.random.uniform(-0.0010, 0.0010, num_bars)
    
    tick_volume = np.random.randint(100, 1000, num_bars)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'tick_volume': tick_volume
    }, index=times)
    df.index.name = 'time'
    
    return df

@pytest.fixture
def sample_m15():
    return generate_synthetic_data(300, freq="15min")

@pytest.fixture
def sample_h1():
    return generate_synthetic_data(100, freq="1H")

def test_compute_indicators(sample_m15):
    df_ind = compute_indicators(sample_m15)
    
    # Check if all indicator columns are present
    expected_cols = [
        'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
        'ema_20', 'ema_50', 'ema_ratio',
        'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct',
        'atr_14', 'adx_14'
    ]
    for col in expected_cols:
         assert col in df_ind.columns, f"{col} missing in output"
         
    # Check that tail has no NaNs for these indicators
    assert not df_ind[expected_cols].iloc[-10:].isna().any().any()

def test_compute_session_features(sample_m15):
    df_sess = compute_session_features(sample_m15)
    
    expected_cols = [
        'is_london', 'is_ny', 'is_overlap',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    for col in expected_cols:
        assert col in df_sess.columns, f"{col} missing in output"
        
    # Test valid range of values
    assert df_sess['is_london'].isin([0, 0.5]).all()
    assert df_sess['is_ny'].isin([0, 0.5]).all()
    assert df_sess['is_overlap'].isin([0, 0.5]).all()

def test_add_h1_trend(sample_m15, sample_h1):
    df_trend = add_h1_trend(sample_m15, sample_h1)
    
    assert 'h1_trend' in df_trend.columns
    # Check that values are +1 or -1 where capable
    # Initially there might be some -1 or +1. 
    unique_vals = set(df_trend['h1_trend'].dropna().unique())
    assert unique_vals.issubset({1, -1})

def test_build_feature_matrix(sample_m15, sample_h1):
    df_feats = build_feature_matrix(sample_m15, sample_h1)
    
    # Assert zero NaN in output
    assert df_feats.isna().sum().sum() == 0, "There are NaNs in the feature matrix output"
    
    # Assert output shape and features match
    assert list(df_feats.columns) == FEATURE_COLS
    
    # No raw OHLCV
    for raw in ['open', 'high', 'low', 'close', 'tick_volume', 'volume']:
        assert raw not in df_feats.columns

@patch('features.get_ohlcv')
def test_get_live_features(mock_get_hist, sample_m15, sample_h1):
    def side_effect(symbol, timeframe_str, count):
        if timeframe_str == "M15":
            return sample_m15.iloc[-count:]
        elif timeframe_str == "H1":
            return sample_h1.iloc[-count:]
        return pd.DataFrame()
        
    mock_get_hist.side_effect = side_effect
    
    features_series = get_live_features("EURUSD")
    
    assert isinstance(features_series, pd.Series)
    assert len(features_series) == len(FEATURE_COLS)
    assert not features_series.isna().any()
