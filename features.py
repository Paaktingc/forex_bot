"""
features.py

Feature engineering module. Uses pandas-ta to compute technical indicators,
session embeddings, and multi-timeframe trends.
"""

import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
import MetaTrader5 as mt5

import config
from data_feed import get_ohlcv

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'ema_20', 'ema_50', 'ema_ratio',
    'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct',
    'atr_14', 'adx_14',
    'is_london', 'is_ny', 'is_overlap',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'h1_trend'
]

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators using pandas-ta.
    """
    df_out = df.copy()
    
    # pandas-ta requires 'volume' instead of 'tick_volume' for some indicators if needed
    if 'tick_volume' in df_out.columns and 'volume' not in df_out.columns:
        df_out['volume'] = df_out['tick_volume']
        
    # RSI(14)
    df_out.ta.rsi(length=14, append=True)
    df_out.rename(columns={'RSI_14': 'rsi_14'}, inplace=True)
    
    # MACD(12, 26, 9)
    df_out.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_out.rename(columns={
        'MACD_12_26_9': 'macd_line',
        'MACDh_12_26_9': 'macd_hist',
        'MACDs_12_26_9': 'macd_signal'
    }, inplace=True)
    
    # EMA 20 and 50
    df_out.ta.ema(length=20, append=True)
    df_out.rename(columns={'EMA_20': 'ema_20'}, inplace=True)
    
    df_out.ta.ema(length=50, append=True)
    df_out.rename(columns={'EMA_50': 'ema_50'}, inplace=True)
    
    # EMA Ratio
    df_out['ema_ratio'] = df_out['ema_20'] / df_out['ema_50']
    
    # Bollinger Bands(20, 2)
    df_out.ta.bbands(length=20, std=2, append=True)
    
    # Safely rename Bollinger Bands columns due to pandas-ta returning varied suffix
    bb_lower_col = next(c for c in df_out.columns if c.startswith('BBL_20'))
    bb_mid_col = next(c for c in df_out.columns if c.startswith('BBM_20'))
    bb_upper_col = next(c for c in df_out.columns if c.startswith('BBU_20'))
    
    df_out.rename(columns={
        bb_lower_col: 'bb_lower',
        bb_mid_col: 'bb_mid',
        bb_upper_col: 'bb_upper'
    }, inplace=True)
    
    # Override bb_pct calculation: (close - bb_lower) / (bb_upper - bb_lower)
    df_out['bb_pct'] = (df_out['close'] - df_out['bb_lower']) / (df_out['bb_upper'] - df_out['bb_lower'])
    
    # ATR(14)
    df_out.ta.atr(length=14, append=True)
    df_out.rename(columns={'ATRr_14': 'atr_14'}, inplace=True)
    
    # ADX(14)
    df_out.ta.adx(length=14, append=True)
    df_out.rename(columns={'ADX_14': 'adx_14'}, inplace=True)
    
    return df_out

def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes session flags and cyclical time features based on config.
    Assumes DataFrame has a DatetimeIndex.
    """
    df_out = df.copy()
    
    hours = df_out.index.hour
    dow = df_out.index.dayofweek
    
    df_out['is_london'] = ((hours >= config.LONDON_START_UTC) & (hours <= config.LONDON_END_UTC)).astype(int)
    df_out['is_ny'] = ((hours >= config.NY_START_UTC) & (hours <= config.NY_END_UTC)).astype(int)
    
    # is_overlap: 1 if both london and ny active (13–16 UTC)
    # Using the specific prompt definition explicitly, or the intersection
    df_out['is_overlap'] = ((hours >= 13) & (hours <= 16)).astype(int)
    
    # Cyclical hour and day of week
    df_out['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_out['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    
    df_out['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df_out['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    return df_out

def add_h1_trend(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns H1 EMA(50) backwards to M15 timeframe to determine trend.
    +1 if H1 close > H1 EMA50, else -1.
    """
    df_h1_out = df_h1.copy()
    # Compute EMA(50) on H1 data
    df_h1_out.ta.ema(length=50, append=True)
    
    # If EMA_50 column doesn't exist (too few bars), create it from close
    if 'EMA_50' not in df_h1_out.columns:
        df_h1_out['EMA_50'] = df_h1_out['close']
    
    m15_reset = df_m15.reset_index()
    h1_reset = df_h1_out.reset_index()
    
    # Ensure time columns exist
    if 'time' not in m15_reset.columns and 'index' in m15_reset.columns:
        m15_reset = m15_reset.rename(columns={'index': 'time'})
    if 'time' not in h1_reset.columns and 'index' in h1_reset.columns:
        h1_reset = h1_reset.rename(columns={'index': 'time'})
        
    merged = pd.merge_asof(
        m15_reset.sort_values('time'),
        h1_reset[['time', 'close', 'EMA_50']].sort_values('time'),
        on='time',
        direction='backward',
        suffixes=('', '_h1')
    )
    
    # Add h1_trend: +1 if H1 close > H1 EMA50, else -1
    merged['h1_trend'] = np.where(merged['close_h1'] > merged['EMA_50'], 1, -1)
    
    merged.set_index('time', inplace=True)
    return merged

def build_feature_matrix(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Sequentially builds the full feature matrix, dropping NaNs and raw columns.
    """
    if df_m15.empty or df_h1.empty:
        return pd.DataFrame(columns=FEATURE_COLS)
        
    df = compute_indicators(df_m15)
    df = compute_session_features(df)
    df = add_h1_trend(df, df_h1)
    
    # Drop rows with any NaN
    df.dropna(inplace=True)
    
    # Return only feature columns (No raw OHLCV)
    # Ensure we only pick existent columns just in case, but they should all exist
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in feature matrix: {missing_cols}")
        
    cols_to_keep = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols_to_keep]

def get_live_features(symbol: str) -> pd.Series:
    """
    Fetches live M15 and H1 data, builds feature matrix, and returns the current candle features.
    """
    df_m15 = get_ohlcv(symbol, "M15", 200)
    df_h1 = get_ohlcv(symbol, "H1", 100)
    
    if df_m15 is None or df_m15.empty or df_h1 is None or df_h1.empty:
        logger.error(f"Failed to fetch sufficient data for {symbol}")
        return pd.Series(dtype=float)
        
    df_feats = build_feature_matrix(df_m15, df_h1)
    
    if df_feats.empty:
        logger.error(f"Feature matrix is empty for {symbol} after computing indicators")
        return pd.Series(dtype=float)
        
    # Return last row as Series
    return df_feats.iloc[-1]
