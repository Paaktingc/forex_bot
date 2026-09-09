"""
Frozen volatility-expansion continuation signal (Gate 2).

Causality: every quantity at bar t uses only bars <= t, and the ATR reference
that classifies bar t excludes bar t itself (ATR through t-1). The decision is
taken at the close of the expansion bar; entry is the NEXT bar's open (handled
in backtest.py).

Parameters are frozen in config.py and must not be tuned after seeing results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .prepare_data import add_session_flag


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr_reference(df: pd.DataFrame, length: int = C.ATR_LEN) -> pd.Series:
    """ATR(length) computed THROUGH t-1 (SMA of true range, shifted by one bar
    so the classified bar is excluded from its own reference)."""
    tr = true_range(df)
    return tr.rolling(length, min_periods=length).mean().shift(1)


def raw_signal_frame(df: pd.DataFrame, *, tf: str = C.PRIMARY_TF,
                     expansion_mult: float = C.EXPANSION_MULT,
                     stop_atr: float = C.STOP_ATR) -> pd.DataFrame:
    """Per-bar signal components (before cooldown). Direction in {-1,0,+1}."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    tr = true_range(df)
    atr_ref = atr_reference(df)

    rng = (h - l)
    with np.errstate(invalid="ignore", divide="ignore"):
        loc_bull = (c - l) / rng
        loc_bear = (h - c) / rng

    prev_high8 = h.rolling(C.BREAKOUT_LOOKBACK).max().shift(1)
    prev_low8 = l.rolling(C.BREAKOUT_LOOKBACK).min().shift(1)

    expansion = tr >= expansion_mult * atr_ref
    nonzero_body = c != o
    nonzero_range = rng > 0

    bull = (expansion & nonzero_body & nonzero_range & (c > o)
            & (loc_bull >= C.CLOSE_LOCATION) & (c > prev_high8))
    bear = (expansion & nonzero_body & nonzero_range & (c < o)
            & (loc_bear >= C.CLOSE_LOCATION) & (c < prev_low8))

    direction = pd.Series(0, index=df.index, dtype=int)
    direction[bull.fillna(False)] = 1
    direction[bear.fillna(False)] = -1

    session = add_session_flag(df, tf)
    direction[~session] = 0

    return pd.DataFrame({
        "direction": direction,
        "atr_ref": atr_ref,
        "tr": tr,
        "stop_dist": stop_atr * atr_ref,
    }, index=df.index)


def accepted_signals(df: pd.DataFrame, *, tf: str = C.PRIMARY_TF,
                     expansion_mult: float = C.EXPANSION_MULT,
                     stop_atr: float = C.STOP_ATR,
                     cooldown_hours: float = C.COOLDOWN_HOURS) -> pd.DataFrame:
    """Apply the rolling cooldown (>= cooldown_hours between accepted signals,
    measured from the signal bar timestamp). Returns accepted signal rows."""
    frame = raw_signal_frame(df, tf=tf, expansion_mult=expansion_mult, stop_atr=stop_atr)
    cand = frame[(frame["direction"] != 0) & frame["stop_dist"].gt(0)
                 & frame["atr_ref"].notna()]
    cd = pd.Timedelta(hours=cooldown_hours)
    accepted_idx = []
    last_ts = None
    for ts in cand.index:
        if last_ts is None or (ts - last_ts) >= cd:
            accepted_idx.append(ts)
            last_ts = ts
    return cand.loc[accepted_idx]
