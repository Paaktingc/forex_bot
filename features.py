"""
features.py

Feature engineering module for technical indicators, session embeddings,
and multi-timeframe trends.
"""

import logging
import pandas as pd
import numpy as np

import config
from data_feed import get_ohlcv

logger = logging.getLogger(__name__)

H1_FEATURE_COLS = [
    "atr_14",
    "ema_20",
    "ema_50",
    "ema_200",
    "ema_20_50_diff",
    "close_ema20_diff",
    "close_ema50_diff",
    "rsi_14",
    "adx_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_pct",
    "ret_1",
    "ret_4",
    "ret_24",
    "vol_ratio",
]

FEATURE_COLS = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'ema_20', 'ema_50', 'ema_ratio',
    'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct',
    'atr_14', 'adx_14',
    'price_momentum', 'volatility_ratio', 'bb_position', 'macd_cross', 'rsi_extreme',
    'is_london', 'is_ny', 'is_overlap',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'h1_trend'
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr = _atr(df, length)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _validate_h1_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise an H1 OHLCV frame.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    out = df.copy()
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out["volume"] = out["tick_volume"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must use a DatetimeIndex.")

    return out.sort_index()


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def feature_engineering_h1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute H1 features using ONLY data at time ``t`` and earlier.

    CAUSALITY AUDIT
    - ``atr_14`` uses high/low/close up to and including bar ``t``
    - ``ema_20`` / ``ema_50`` / ``ema_200`` use closes up to ``t``
    - ``ema_20_50_diff`` is derived from current EMAs only
    - ``close_ema20_diff`` / ``close_ema50_diff`` use ``close[t]`` only
    - ``rsi_14`` uses closes up to ``t``
    - ``adx_14`` uses high/low/close up to ``t``
    - ``macd`` / ``macd_signal`` / ``macd_hist`` use closes up to ``t``
    - ``bb_width`` / ``bb_pct`` use rolling windows ending at ``t``
    - ``ret_1`` / ``ret_4`` / ``ret_24`` use trailing returns ending at ``t``
    - ``vol_ratio`` uses trailing rolling volatility ending at ``t``

    Forbidden and intentionally absent:
    - forward returns such as ``close[t+1] / close[t]``
    - centred rolling windows
    - any feature derived from a target column
    - any feature using ``open[t+1]``

    The returned frame keeps OHLCV plus engineered features. Warmup rows with
    incomplete indicators are dropped.
    """
    out = _validate_h1_frame(df)
    close = out["close"]
    atr = _atr(out, 14)
    atr_safe = atr.replace(0, np.nan)

    out["atr_14"] = atr
    out["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema_200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema_20_50_diff"] = (out["ema_20"] - out["ema_50"]) / atr_safe
    out["close_ema20_diff"] = (close - out["ema_20"]) / atr_safe
    out["close_ema50_diff"] = (close - out["ema_50"]) / atr_safe
    out["rsi_14"] = _rsi_wilder(close, 14)
    out["adx_14"] = _adx(out, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_width"] = (2.0 * std20) / sma20.replace(0, np.nan)
    out["bb_pct"] = (close - (sma20 - 2.0 * std20)) / (4.0 * std20).replace(0, np.nan)

    out["ret_1"] = close.pct_change(1)
    out["ret_4"] = close.pct_change(4)
    out["ret_24"] = close.pct_change(24)

    short_vol = close.pct_change().rolling(5, min_periods=5).std()
    long_vol = close.pct_change().rolling(20, min_periods=20).std().replace(0, np.nan)
    out["vol_ratio"] = short_vol / long_vol

    out = out.dropna(subset=H1_FEATURE_COLS).copy()
    return out

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators using pandas/numpy only.
    """
    df_out = df.copy()
    
    if 'tick_volume' in df_out.columns and 'volume' not in df_out.columns:
        df_out['volume'] = df_out['tick_volume']

    close = df_out["close"]
    df_out["rsi_14"] = _rsi(close, 14)

    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    df_out["macd_line"] = ema_fast - ema_slow
    df_out["macd_signal"] = df_out["macd_line"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    df_out["macd_hist"] = df_out["macd_line"] - df_out["macd_signal"]

    df_out["ema_20"] = _ema(close, 20)
    df_out["ema_50"] = _ema(close, 50)
    df_out["ema_ratio"] = df_out["ema_20"] / df_out["ema_50"]

    bb_mid = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    df_out["bb_mid"] = bb_mid
    df_out["bb_upper"] = bb_mid + 2 * bb_std
    df_out["bb_lower"] = bb_mid - 2 * bb_std
    band_width = (df_out["bb_upper"] - df_out["bb_lower"]).replace(0, np.nan)
    df_out["bb_pct"] = (close - df_out["bb_lower"]) / band_width

    df_out["atr_14"] = _atr(df_out, 14)
    df_out["adx_14"] = _adx(df_out, 14)
    df_out["price_momentum"] = (close - close.shift(20)) / close.shift(20)
    df_out["volatility_ratio"] = df_out["atr_14"] / close.rolling(50, min_periods=50).mean()
    df_out["bb_position"] = (close - df_out["bb_lower"]) / band_width
    df_out["macd_cross"] = (
        (df_out["macd_hist"] > 0) & (df_out["macd_hist"].shift(1) <= 0)
    ).astype(int)
    df_out["rsi_extreme"] = (
        (df_out["rsi_14"] < 30) | (df_out["rsi_14"] > 70)
    ).astype(int)
    
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
    df_out['is_london'] *= 0.5
    df_out['is_ny'] *= 0.5
    df_out['is_overlap'] *= 0.5
    
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
    df_h1_out["EMA_50"] = _ema(df_h1_out["close"], 50)
    if df_h1_out["EMA_50"].isna().all():
        df_h1_out['EMA_50'] = df_h1_out['close']
    
    m15_reset = df_m15.reset_index()
    h1_reset = df_h1_out.reset_index()
    
    # Normalize reset-index datetime column names for merge_asof.
    if 'time' not in m15_reset.columns:
        first_col = m15_reset.columns[0]
        m15_reset = m15_reset.rename(columns={first_col: 'time'})
    if 'time' not in h1_reset.columns:
        first_col = h1_reset.columns[0]
        h1_reset = h1_reset.rename(columns={first_col: 'time'})
        
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
