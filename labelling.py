"""
labelling.py

Causally clean target generation for the EURUSD research pipeline plus legacy
helpers still used by the original bot codepaths.
"""

from __future__ import annotations

import logging
import pickle
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config
from data_feed import get_ohlcv_from_csv
from features import FEATURE_COLS, build_feature_matrix

logger = logging.getLogger(__name__)

REQUIRED_OHLCV = ("open", "high", "low", "close")
TRADE_DIRECTIONS = {"long", "short"}


def _validate_ohlc_frame(df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_OHLCV) -> None:
    """Raise a helpful error when the input frame is not usable for labelling."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range using Wilder's method.
    """
    _validate_ohlc_frame(df)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ADX using Wilder's smoothing.
    """
    _validate_ohlc_frame(df)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    atr_smooth = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean() / atr_smooth.replace(0, np.nan)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def regime_filter(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.Series:
    """
    Return +1 for uptrend, -1 for downtrend, 0 for range.

    Logic:
      - ADX > threshold AND close > EMA_fast > EMA_slow -> +1
      - ADX > threshold AND close < EMA_fast < EMA_slow -> -1
      - Otherwise -> 0
    """
    _validate_ohlc_frame(df)

    close = df["close"]
    adx = compute_adx(df, period=adx_period)
    ema_f = close.ewm(span=ema_fast, min_periods=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, min_periods=ema_slow, adjust=False).mean()

    labels = pd.Series(0, index=df.index, dtype=int)
    up_mask = (adx > adx_threshold) & (close > ema_f) & (ema_f > ema_s)
    down_mask = (adx > adx_threshold) & (close < ema_f) & (ema_f < ema_s)
    labels.loc[up_mask] = 1
    labels.loc[down_mask] = -1

    nan_mask = adx.isna() | ema_f.isna() | ema_s.isna()
    labels.loc[nan_mask] = 0
    return labels


def regime_labels(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.Series:
    """
    Backwards-compatible alias for the old regime label helper.

    The regime state is now treated as a feature, not a training target.
    """
    return regime_filter(
        df,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
    )


def verify_label_distribution(
    labels: pd.Series,
    expected_range: tuple[float, float] = (0.55, 0.85),
    expected_trend: tuple[float, float] = (0.07, 0.25),
) -> dict[str, object]:
    """
    Verify that regime labels are in a healthy range for H1 EURUSD training.
    """
    if labels is None or labels.empty:
        raise ValueError("labels must be a non-empty Series")

    counts = labels.value_counts(normalize=True)
    distribution = {
        "uptrend_pct": float(counts.get(1, 0.0) * 100.0),
        "downtrend_pct": float(counts.get(-1, 0.0) * 100.0),
        "range_pct": float(counts.get(0, 0.0) * 100.0),
        "total_bars": int(len(labels)),
        "nan_count": int(labels.isna().sum()),
    }

    range_ok = expected_range[0] <= distribution["range_pct"] / 100.0 <= expected_range[1]
    up_ok = expected_trend[0] <= distribution["uptrend_pct"] / 100.0 <= expected_trend[1]
    down_ok = expected_trend[0] <= distribution["downtrend_pct"] / 100.0 <= expected_trend[1]

    diagnosis: list[str] = []
    if not range_ok:
        diagnosis.append(
            f"Range class {distribution['range_pct']:.1f}% outside "
            f"[{expected_range[0] * 100:.0f}%, {expected_range[1] * 100:.0f}%]"
        )
    if not up_ok:
        diagnosis.append(
            f"Uptrend class {distribution['uptrend_pct']:.1f}% outside "
            f"[{expected_trend[0] * 100:.0f}%, {expected_trend[1] * 100:.0f}%]"
        )
    if not down_ok:
        diagnosis.append(
            f"Downtrend class {distribution['downtrend_pct']:.1f}% outside "
            f"[{expected_trend[0] * 100:.0f}%, {expected_trend[1] * 100:.0f}%]"
        )

    distribution["healthy"] = range_ok and up_ok and down_ok
    distribution["diagnosis"] = diagnosis
    return distribution


def _validate_target_frame(df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
    _validate_ohlc_frame(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Trade target generation requires a DatetimeIndex.")
    if atr_col not in df.columns:
        raise KeyError(f"Missing ATR column: {atr_col}")
    return df.sort_index().copy()


def _compute_trade_target_array(
    *,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    atr_values: np.ndarray,
    tp_atr_mult: float,
    sl_atr_mult: float,
    max_holding_bars: int,
    direction: Literal["long", "short"],
) -> np.ndarray:
    labels = np.full(len(open_prices), np.nan, dtype=float)
    last_valid_index = len(open_prices) - max_holding_bars
    if last_valid_index <= 0:
        return labels

    for idx in range(last_valid_index):
        atr_value = atr_values[idx]
        entry_idx = idx + 1
        entry_price = open_prices[entry_idx]
        if np.isnan(atr_value) or atr_value <= 0 or np.isnan(entry_price):
            continue

        if direction == "long":
            take_profit = entry_price + tp_atr_mult * atr_value
            stop_loss = entry_price - sl_atr_mult * atr_value
        else:
            take_profit = entry_price - tp_atr_mult * atr_value
            stop_loss = entry_price + sl_atr_mult * atr_value

        outcome = 0.0
        scan_stop = min(entry_idx + max_holding_bars, len(open_prices))
        for future_idx in range(entry_idx, scan_stop):
            high_value = high_prices[future_idx]
            low_value = low_prices[future_idx]
            if np.isnan(high_value) or np.isnan(low_value):
                continue

            if direction == "long":
                hit_tp = high_value >= take_profit
                hit_sl = low_value <= stop_loss
            else:
                hit_tp = low_value <= take_profit
                hit_sl = high_value >= stop_loss

            if hit_tp and hit_sl:
                outcome = 0.0
                break
            if hit_tp:
                outcome = 1.0
                break
            if hit_sl:
                outcome = 0.0
                break

        labels[idx] = outcome

    return labels


def compute_trade_target(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    max_holding_bars: int = 24,
    direction: str = "long",
) -> pd.Series:
    """
    Binary trade outcome target with strict next-bar execution.

    Features at bar ``t`` may only use data available by the close of ``t``.
    The target for bar ``t`` therefore starts at the next bar:
      - entry = open[t+1]
      - TP/SL distance = ATR[t] multipliers
      - future scan window = bars [t+1, t+1+max_holding_bars)

    The last ``max_holding_bars`` rows are always NaN because the future path is
    incomplete.
    """
    clean = _validate_target_frame(df, atr_col)
    if max_holding_bars <= 0:
        raise ValueError("max_holding_bars must be positive.")
    if tp_atr_mult <= 0 or sl_atr_mult <= 0:
        raise ValueError("TP/SL ATR multipliers must be positive.")
    if direction not in TRADE_DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(TRADE_DIRECTIONS)}")

    labels = _compute_trade_target_array(
        open_prices=clean["open"].to_numpy(dtype=float),
        high_prices=clean["high"].to_numpy(dtype=float),
        low_prices=clean["low"].to_numpy(dtype=float),
        atr_values=clean[atr_col].to_numpy(dtype=float),
        tp_atr_mult=float(tp_atr_mult),
        sl_atr_mult=float(sl_atr_mult),
        max_holding_bars=int(max_holding_bars),
        direction=direction,
    )
    return pd.Series(labels, index=clean.index, name=f"target_{direction}")


def generate_directional_targets(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    max_holding_bars: int = 24,
) -> pd.DataFrame:
    """
    Generate causally clean long/short trade outcomes for every bar.

    ``target_best`` answers:
      - ``+1`` if the long trade has edge and the short does not
      - ``-1`` if the short trade has edge and the long does not
      - ``0`` if both lose or both win
      - ``NaN`` if there is not enough future data to evaluate the trade
    """
    clean = _validate_target_frame(df, atr_col)
    target_long = compute_trade_target(
        clean,
        atr_col=atr_col,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        max_holding_bars=max_holding_bars,
        direction="long",
    )
    target_short = compute_trade_target(
        clean,
        atr_col=atr_col,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        max_holding_bars=max_holding_bars,
        direction="short",
    )

    target_best = pd.Series(np.nan, index=clean.index, name="target_best", dtype=float)
    valid_mask = target_long.notna() & target_short.notna()
    target_best.loc[valid_mask] = 0.0
    target_best.loc[valid_mask & (target_long == 1.0) & (target_short == 0.0)] = 1.0
    target_best.loc[valid_mask & (target_short == 1.0) & (target_long == 0.0)] = -1.0

    return pd.DataFrame(
        {
            "target_long": target_long,
            "target_short": target_short,
            "target_best": target_best,
        },
        index=clean.index,
    )


def apply_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the triple-barrier method to generate target labels.

    Barriers:
    1. Upper: entry + (TRIPLE_BARRIER_UPPER_MULT * ATR_14)
    2. Lower: entry - (TRIPLE_BARRIER_LOWER_MULT * ATR_14)
    3. Time: Next TRIPLE_BARRIER_TIME_LIMIT candles
    """
    df = df.copy()
    if df.empty or "close" not in df.columns or "atr_14" not in df.columns:
        logger.warning("Missing required columns ('close', 'atr_14') or DataFrame empty.")
        return df

    upper_mult = config.TRIPLE_BARRIER_UPPER_MULT
    lower_mult = config.TRIPLE_BARRIER_LOWER_MULT
    time_limit = config.TRIPLE_BARRIER_TIME_LIMIT

    labels = np.zeros(len(df), dtype=int)
    closes = df["close"].values
    atrs = df["atr_14"].values

    for i in range(len(df) - time_limit):
        entry = closes[i]
        atr = atrs[i]

        if pd.isna(atr) or atr <= 0:
            labels[i] = 0
            continue

        upper = entry + upper_mult * atr
        lower = entry - lower_mult * atr

        label = 0
        for j in range(i + 1, i + 1 + time_limit):
            future_close = closes[j]
            if future_close >= upper:
                label = 1
                break
            if future_close <= lower:
                label = -1
                break
        labels[i] = label

    df["label"] = labels
    return df.iloc[:-time_limit].copy()


def get_label_distribution(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """
    Returns the count and percentage of -1, 0, 1 labels.
    """
    if "label" not in df.columns:
        return {}

    counts = df["label"].value_counts().to_dict()
    total = len(df)

    distribution: dict[int, dict[str, float]] = {}
    for label_val in (-1, 0, 1):
        count = counts.get(label_val, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        distribution[label_val] = {"count": int(count), "percentage": float(pct)}

    return distribution


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Extracts features and target, encodes labels, and downsamples the majority
    class (0) if it is extremely dominant.
    """
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning("Missing FEATURE_COLS in df: %s", missing_cols)

    valid_feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df.dropna(subset=valid_feature_cols + ["label"]).copy()

    X = df_clean[valid_feature_cols]
    y = df_clean["label"].values

    counts = pd.Series(y).value_counts()
    count_0 = counts.get(0, 0)
    count_others = max(1, len(y) - count_0)
    if count_0 / count_others > 10:
        logger.info("Significant class imbalance detected. Downsampling class 0.")
        idx_0 = np.where(y == 0)[0]
        idx_others = np.where(y != 0)[0]
        np.random.seed(42)
        target_0_count = min(count_others * 10, len(idx_0))
        idx_0_sampled = np.random.choice(idx_0, size=target_0_count, replace=False)
        selected_idx = np.sort(np.concatenate([idx_0_sampled, idx_others]))
        X = X.iloc[selected_idx]
        y = y[selected_idx]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le


def save_prepared_data(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder) -> None:
    """
    Saves features, target, and the label encoder to disk.
    """
    X.to_csv(config.DATA_DIR / "X_train.csv", index=True)
    np.save(config.DATA_DIR / "y_train.npy", y)

    with open(config.ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    logger.info("Saved prepared data and encoder to %s and %s", config.DATA_DIR, config.MODELS_DIR)


def _set_barrier_config(upper: float, lower: float, time_limit: int) -> None:
    config.TRIPLE_BARRIER_UPPER_MULT = float(upper)
    config.TRIPLE_BARRIER_LOWER_MULT = float(lower)
    config.TRIPLE_BARRIER_TIME_LIMIT = int(time_limit)


def _load_real_feature_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_m15 = get_ohlcv_from_csv(config.SYMBOL, config.TIMEFRAME_PRIMARY)
    df_h1 = get_ohlcv_from_csv(config.SYMBOL, config.TIMEFRAME_TREND)
    feature_frame = build_feature_matrix(df_m15, df_h1)
    if feature_frame.empty:
        raise ValueError("Feature matrix is empty; cannot continue labelling.")
    if feature_frame.isnull().sum().sum() != 0:
        raise ValueError("Feature matrix contains NaN values.")

    missing_cols = [col for col in FEATURE_COLS if col not in feature_frame.columns]
    if missing_cols:
        raise ValueError(f"Feature matrix missing FEATURE_COLS: {missing_cols}")

    feature_frame.to_csv(config.DATA_DIR / "EURUSD_features.csv")
    return df_m15, feature_frame


def _join_prices(feature_frame: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    raw_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in raw_cols if col not in df_m15.columns]
    if missing_cols:
        raise ValueError(f"M15 price data missing required columns: {missing_cols}")

    joined = df_m15.loc[feature_frame.index, raw_cols].copy()
    for column in feature_frame.columns:
        joined[column] = feature_frame[column]
    return joined


def _distribution_within_secondary_range(distribution: dict[int, dict[str, float]]) -> bool:
    return all(25.0 <= distribution[label]["percentage"] <= 40.0 for label in (-1, 0, 1))


def _print_distribution(distribution: dict[int, dict[str, float]], header: str) -> None:
    print(header)
    print(
        f"Label -1 (sell): {distribution[-1]['count']} "
        f"({distribution[-1]['percentage']:.2f}%)"
    )
    print(
        f"Label  0 (hold): {distribution[0]['count']} "
        f"({distribution[0]['percentage']:.2f}%)"
    )
    print(
        f"Label  1 (buy):  {distribution[1]['count']} "
        f"({distribution[1]['percentage']:.2f}%)"
    )


def run_labelling_pipeline() -> dict[str, object]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    base_attempt = (
        float(config.TRIPLE_BARRIER_UPPER_MULT),
        float(config.TRIPLE_BARRIER_LOWER_MULT),
        int(config.TRIPLE_BARRIER_TIME_LIMIT),
    )
    attempts = [base_attempt]
    final_distribution: dict[int, dict[str, float]] | None = None
    final_X: pd.DataFrame | None = None
    final_y: np.ndarray | None = None
    final_le: LabelEncoder | None = None
    final_params = base_attempt

    df_m15, feature_frame = _load_real_feature_frame()
    price_joined = _join_prices(feature_frame, df_m15)

    attempt_index = 0
    while attempt_index < len(attempts):
        upper, lower, time_limit = attempts[attempt_index]
        _set_barrier_config(upper, lower, time_limit)
        labelled = apply_triple_barrier(price_joined)
        distribution = get_label_distribution(labelled)
        _print_distribution(
            distribution,
            header=(
                f"\nAttempt {attempt_index + 1} "
                f"(upper={upper}, lower={lower}, time_limit={time_limit})"
            ),
        )

        hold_pct = distribution[0]["percentage"]
        if attempt_index == 0:
            if hold_pct < 20.0:
                retry_attempt = (0.9, 0.6, time_limit)
                if retry_attempt != attempts[0]:
                    attempts.append(retry_attempt)
            elif hold_pct > 50.0:
                retry_attempt = (1.1, 0.8, time_limit)
                if retry_attempt != attempts[0]:
                    attempts.append(retry_attempt)

        final_distribution = distribution
        final_params = (upper, lower, time_limit)
        final_X, final_y, final_le = prepare_training_data(labelled)
        attempt_index += 1

    if final_X is None or final_y is None or final_le is None or final_distribution is None:
        raise RuntimeError("Labelling pipeline did not produce training artifacts.")

    save_prepared_data(final_X, final_y, final_le)
    _set_barrier_config(*final_params)

    if not _distribution_within_secondary_range(final_distribution):
        print(
            "Final label mix remains outside the 25-40% secondary balance range; "
            "continuing with the best bounded retry result."
        )

    print("\nEncoded class mapping")
    for encoded_value, original_label in enumerate(final_le.classes_):
        print(f"{encoded_value} -> {original_label}")

    print(f"\nFinal dataset shape: X={final_X.shape[0]}x{final_X.shape[1]}, y={len(final_y)}")
    print(
        f"Saved: {config.DATA_DIR / 'EURUSD_features.csv'} | "
        f"{config.DATA_DIR / 'X_train.csv'} | {config.DATA_DIR / 'y_train.npy'} | "
        f"{config.ENCODER_PATH}"
    )
    return {
        "distribution": final_distribution,
        "shape": final_X.shape,
        "barrier_params": {
            "upper": final_params[0],
            "lower": final_params[1],
            "time_limit": final_params[2],
        },
    }


if __name__ == "__main__":
    run_labelling_pipeline()
