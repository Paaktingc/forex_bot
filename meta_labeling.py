"""
meta_labeling.py

Meta-labeling utilities for filtering rule-based base signals.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from labelling import compute_adx, generate_directional_targets


def _validate_signal_frame(df: pd.DataFrame, atr_col: str = "atr_14") -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    required = ["open", "high", "low", "close", atr_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required signal columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Signal generation requires a DatetimeIndex.")
    return df.sort_index().copy()


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray | None:
    if len(y) == 0:
        return None
    labels, counts = np.unique(y, return_counts=True)
    if float(counts.max() / counts.sum()) <= 0.60:
        return None
    weights = {label: len(y) / (len(labels) * count) for label, count in zip(labels, counts, strict=False)}
    return np.array([weights[label] for label in y], dtype=float)


def generate_base_signals(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.Series:
    """
    Rule-based trend-following entry signals using only information available at bar ``t``.
    """
    clean = _validate_signal_frame(df)
    close = clean["close"]
    adx = compute_adx(clean, period=14)
    ema_fast_series = close.ewm(span=ema_fast, min_periods=ema_fast, adjust=False).mean()
    ema_slow_series = close.ewm(span=ema_slow, min_periods=ema_slow, adjust=False).mean()

    signals = pd.Series(0, index=clean.index, dtype=int)
    buy_mask = (adx > adx_threshold) & (close > ema_fast_series) & (ema_fast_series > ema_slow_series)
    sell_mask = (adx > adx_threshold) & (close < ema_fast_series) & (ema_fast_series < ema_slow_series)
    signals.loc[buy_mask] = 1
    signals.loc[sell_mask] = -1
    signals.loc[adx.isna() | ema_fast_series.isna() | ema_slow_series.isna()] = 0
    return signals


def label_signal_outcomes(
    df: pd.DataFrame,
    signals: pd.Series,
    atr_col: str = "atr_14",
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    max_holding_bars: int = 24,
) -> pd.Series:
    """
    Label each non-zero base signal as profitable (1) or losing (0).
    """
    clean = _validate_signal_frame(df, atr_col=atr_col)
    aligned_signals = signals.reindex(clean.index).fillna(0).astype(int)
    targets = generate_directional_targets(
        clean,
        atr_col=atr_col,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        max_holding_bars=max_holding_bars,
    )

    outcomes = pd.Series(np.nan, index=clean.index, name="signal_outcome", dtype=float)
    buy_mask = aligned_signals == 1
    sell_mask = aligned_signals == -1
    outcomes.loc[buy_mask] = targets.loc[buy_mask, "target_long"]
    outcomes.loc[sell_mask] = targets.loc[sell_mask, "target_short"]
    return outcomes


def train_meta_model(
    df: pd.DataFrame,
    signals: pd.Series,
    outcomes: pd.Series,
    feature_cols: List[str],
    train_mask: pd.Series,
) -> object:
    """
    Train a shallower secondary classifier that predicts whether a base signal will win.
    """
    clean = _validate_signal_frame(df)
    aligned_signals = signals.reindex(clean.index).fillna(0).astype(int)
    aligned_outcomes = outcomes.reindex(clean.index)
    aligned_train_mask = train_mask.reindex(clean.index).fillna(False).astype(bool)

    eligible = aligned_train_mask & (aligned_signals != 0) & aligned_outcomes.notna()
    if int(eligible.sum()) < 40:
        raise ValueError("Not enough eligible training rows for meta-labeling.")

    X_train = clean.loc[eligible, feature_cols]
    y_train = aligned_outcomes.loc[eligible].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("Meta-labeling training target contains fewer than two classes.")

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=30,
        random_state=42,
    )
    sample_weight = _balanced_sample_weights(y_train)
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def apply_meta_filter(
    model,
    df: pd.DataFrame,
    signals: pd.Series,
    feature_cols: List[str],
    confidence_threshold: float = 0.55,
) -> pd.Series:
    """
    Keep only base signals whose predicted profitability exceeds the threshold.
    """
    clean = _validate_signal_frame(df)
    aligned_signals = signals.reindex(clean.index).fillna(0).astype(int)
    filtered = aligned_signals.copy()
    signal_mask = aligned_signals != 0
    if int(signal_mask.sum()) == 0:
        return filtered

    probabilities = model.predict_proba(clean.loc[signal_mask, feature_cols])[:, 1]
    keep_mask = probabilities >= confidence_threshold
    filtered.loc[signal_mask] = np.where(keep_mask, aligned_signals.loc[signal_mask], 0)
    return filtered.astype(int)
