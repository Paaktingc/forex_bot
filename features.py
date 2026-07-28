"""
features.py

Change summary:
- Rebuilt the research feature set around meta-labeling at signal time.
- Added trend, momentum, volatility, market structure, regime, and time
  features that use only information available by the close of bar t.
- Kept the legacy M15 feature path intact for the original bot modules.

Causality boundary for the research pipeline:
- A signal is generated after H1 bar t has fully closed.
- Every feature below uses only values from bar t and earlier.
- Any trade entry and target evaluation starts at open[t+1].
- No research feature may use shift(-1), centered windows, or future bars.
"""

import logging
import pandas as pd
import numpy as np

import config
from data_feed import get_ohlcv

logger = logging.getLogger(__name__)

META_FEATURE_COLS = [
    "close_to_sma20_atr",
    "close_to_sma50_atr",
    "close_to_sma200_atr",
    "lr_slope_20_atr",
    "rsi_14",
    "roc_5_atr",
    "roc_10_atr",
    "roc_20_atr",
    "fracdiff_close_d04_atr",
    "fracdiff_slope_5",
    "atr_14",
    "atr_20",
    "bb_width_20",
    "atr_20_vov_20",
    "bb_width_vov_20",
    "close_range_pos_20",
    "range_20_atr",
    "adx_14",
    "cusum_break",
    "cusum_direction",
    "bars_since_cusum",
    "prob_high_vol_regime",
    "prob_trend_regime",
    "hour_of_day",
    "day_of_week",
]

H1_FEATURE_COLS = META_FEATURE_COLS

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


def _rolling_linear_slope(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be greater than 1")
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denominator = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        values_mean = values.mean()
        numerator = ((x - x_mean) * (values - values_mean)).sum()
        return float(numerator / denominator) if denominator != 0 else np.nan

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _fracdiff_weights(d: float, threshold: float = 1e-4, max_size: int = 256) -> np.ndarray:
    """
    Fixed-width fractional differentiation weights for ``(1 - B)^d``.

    The weights are deterministic for a chosen d, so this does not calibrate on
    the full dataset. Calibrating d via ADF should happen inside train folds.
    """
    if not 0 < d < 1:
        raise ValueError("d must be between 0 and 1.")
    weights = [1.0]
    for k in range(1, max_size):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(float(weight))
    return np.array(weights, dtype=float)


def _fractional_diff_fixed_width(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-4,
    max_size: int = 256,
) -> pd.Series:
    weights = _fracdiff_weights(d=d, threshold=threshold, max_size=max_size)
    values = series.astype(float).to_numpy()
    output = np.full(len(values), np.nan, dtype=float)
    width = len(weights)

    for row in range(width - 1, len(values)):
        window = values[row - width + 1 : row + 1]
        if np.isnan(window).any():
            continue
        output[row] = float(np.dot(weights, window[::-1]))

    return pd.Series(output, index=series.index, name=f"fracdiff_d{d:g}")


def _symmetric_cusum_events(
    series: pd.Series,
    threshold: pd.Series,
) -> pd.DataFrame:
    """
    Causal symmetric CUSUM events using bar-to-bar price changes.

    A non-zero event at t means the cumulative deviation breached the dynamic
    threshold after observing the completed bar t.
    """
    clean_threshold = threshold.replace(0, np.nan)
    diffs = series.diff()
    events = pd.Series(0, index=series.index, dtype=int)
    bars_since = pd.Series(np.nan, index=series.index, dtype=float)
    pos_sum = 0.0
    neg_sum = 0.0
    last_event_pos: int | None = None

    for pos, timestamp in enumerate(series.index):
        diff = diffs.iloc[pos]
        threshold_value = clean_threshold.iloc[pos]
        if pd.isna(diff) or pd.isna(threshold_value) or threshold_value <= 0:
            continue

        pos_sum = max(0.0, pos_sum + float(diff))
        neg_sum = min(0.0, neg_sum + float(diff))

        event = 0
        if pos_sum > float(threshold_value):
            event = 1
            pos_sum = 0.0
            neg_sum = 0.0
        elif neg_sum < -float(threshold_value):
            event = -1
            pos_sum = 0.0
            neg_sum = 0.0

        if event:
            events.loc[timestamp] = event
            last_event_pos = pos
            bars_since.loc[timestamp] = 0.0
        elif last_event_pos is not None:
            bars_since.loc[timestamp] = float(pos - last_event_pos)

    return pd.DataFrame(
        {
            "cusum_break": (events != 0).astype(int),
            "cusum_direction": events.astype(int),
            "bars_since_cusum": bars_since,
        },
        index=series.index,
    )


def _causal_regime_probability(
    signal: pd.Series,
    lookback: int = 250,
    min_periods: int = 100,
) -> pd.Series:
    """
    Convert a regime-strength signal into a causal rolling probability proxy.

    This is intentionally not an HMM: it uses only past rolling median/IQR and
    can live safely in the feature matrix. A true HMM should be fit inside each
    train fold and then applied to the corresponding test fold.
    """
    median = signal.rolling(lookback, min_periods=min_periods).median()
    q75 = signal.rolling(lookback, min_periods=min_periods).quantile(0.75)
    q25 = signal.rolling(lookback, min_periods=min_periods).quantile(0.25)
    scale = (q75 - q25).replace(0, np.nan)
    z_score = ((signal - median) / scale).clip(-20, 20)
    return 1.0 / (1.0 + np.exp(-z_score))


def feature_engineering_h1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute H1 features using ONLY data at time ``t`` and earlier.

    Decision boundary:
    - the model observes a fully completed H1 bar at timestamp ``t``
    - it decides after that bar closes
    - execution, targets, and any trade simulation begin at ``open[t+1]``
    - therefore features may use OHLCV from the completed bar ``t``, but never
      any value from ``t+1`` or later

    Feature groups:
    - Trend: distances from SMA(20/50/200) in ATR units, plus 20-bar regression slope
    - Momentum: RSI(14), 5/10/20-bar price change in ATR units
    - Memory: fixed-d fractional differentiation using a preselected d=0.4
    - Volatility: ATR(14), ATR(20), Bollinger width, volatility-of-volatility
    - Structural breaks: symmetric CUSUM events with ATR-scaled thresholds
    - Market structure: close position in 20-bar range, range size in ATR units
    - Regime: ADX(14), causal continuous high-vol/trend probability proxies
    - Time: hour of day, day of week
    """
    out = _validate_h1_frame(df)
    close = out["close"]
    out["atr_14"] = _atr(out, 14)
    out["atr_20"] = _atr(out, 20)
    out["atr_20_target"] = out["atr_20"]
    atr_safe = out["atr_20"].replace(0, np.nan)

    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["sma_200"] = close.rolling(200, min_periods=200).mean()
    out["close_to_sma20_atr"] = (close - out["sma_20"]) / atr_safe
    out["close_to_sma50_atr"] = (close - out["sma_50"]) / atr_safe
    out["close_to_sma200_atr"] = (close - out["sma_200"]) / atr_safe
    out["lr_slope_20_atr"] = _rolling_linear_slope(close, 20) / atr_safe
    out["rsi_14"] = _rsi_wilder(close, 14)
    out["adx_14"] = _adx(out, 14)

    out["roc_5_atr"] = (close - close.shift(5)) / atr_safe
    out["roc_10_atr"] = (close - close.shift(10)) / atr_safe
    out["roc_20_atr"] = (close - close.shift(20)) / atr_safe
    fracdiff_close = _fractional_diff_fixed_width(close, d=0.4)
    out["fracdiff_close_d04_atr"] = fracdiff_close / atr_safe
    out["fracdiff_slope_5"] = fracdiff_close.diff(5) / atr_safe

    sma20 = out["sma_20"]
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_width_20"] = std20 / sma20.replace(0, np.nan)
    out["atr_20_vov_20"] = out["atr_20"].pct_change().rolling(20, min_periods=20).std()
    out["bb_width_vov_20"] = out["bb_width_20"].pct_change().rolling(20, min_periods=20).std()

    rolling_high_20 = out["high"].rolling(20, min_periods=20).max()
    rolling_low_20 = out["low"].rolling(20, min_periods=20).min()
    range_20 = (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    out["close_range_pos_20"] = (close - rolling_low_20) / range_20
    out["range_20_atr"] = range_20 / atr_safe

    cusum = _symmetric_cusum_events(close, threshold=out["atr_20"] * 1.5)
    out = out.join(cusum)
    out["bars_since_cusum"] = out["bars_since_cusum"].fillna(10_000.0).clip(upper=10_000.0)
    out["prob_high_vol_regime"] = _causal_regime_probability(out["atr_20_vov_20"])
    out["prob_trend_regime"] = _causal_regime_probability(out["adx_14"])

    out["hour_of_day"] = out.index.hour.astype(int)
    out["day_of_week"] = out.index.dayofweek.astype(int)

    out = out.dropna(subset=META_FEATURE_COLS).copy()
    return out

def compute_regime_indicators_h1(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    H1 indicators for the rules-based regime filter:
    EMA(50), EMA(200), ADX(14), ATR(14). Uses only completed-bar data.
    """
    out = df_h1.copy()
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out["volume"] = out["tick_volume"]
    close = out["close"]
    out["ema_50"] = _ema(close, 50)
    out["ema_200"] = _ema(close, 200)
    out["adx_14"] = _adx(out, 14)
    out["atr_14"] = _atr(out, 14)
    return out


def compute_entry_indicators_m15(df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    M15 indicators for the rules-based entry trigger:
    EMA(20), RSI(14), ATR(14). Uses only completed-bar data.
    """
    out = df_m15.copy()
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out["volume"] = out["tick_volume"]
    close = out["close"]
    out["ema_20"] = _ema(close, 20)
    out["rsi_14"] = _rsi_wilder(close, 14)
    out["atr_14"] = _atr(out, 14)
    return out


def rolling_swing_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling swing extremes over the trailing ``window`` completed bars:
    swing_high = max(high), swing_low = min(low). Causal (no future bars).
    """
    return pd.DataFrame(
        {
            "swing_high": df["high"].rolling(window, min_periods=window).max(),
            "swing_low": df["low"].rolling(window, min_periods=window).min(),
        },
        index=df.index,
    )


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
    Aligns H1 EMA(50) onto the M15 timeframe to determine trend.
    +1 if H1 close > H1 EMA50, else -1.

    Leak-free alignment: H1 bars are left-labelled, so their close/EMA are only
    known at label + 1h. align_last_closed_bar attaches the most recently CLOSED
    H1 bar to each M15 timestamp (see htf_alignment / research_log.md Gate 0).
    Previously this used merge_asof(direction="backward") on the H1 label, which
    attached the still-forming H1 bar (future close) — the same structural leak
    that invalidated the H1 regime strategy.
    """
    from htf_alignment import align_last_closed_bar

    df_h1_out = df_h1.copy()
    df_h1_out["EMA_50"] = _ema(df_h1_out["close"], 50)
    if df_h1_out["EMA_50"].isna().all():
        df_h1_out['EMA_50'] = df_h1_out['close']

    aligned = align_last_closed_bar(
        df_m15.index, df_h1_out[["close", "EMA_50"]], columns=["close", "EMA_50"]
    )
    merged = df_m15.copy()
    merged["close_h1"] = aligned["close"].to_numpy()
    merged["EMA_50"] = aligned["EMA_50"].to_numpy()
    merged["h1_trend"] = np.where(merged["close_h1"] > merged["EMA_50"], 1, -1)
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
