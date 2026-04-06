"""
resampler.py

Utilities for standardising lower-timeframe OHLCV data and resampling it to H1.
"""

from __future__ import annotations

import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert common timestamp layouts into a clean, sorted DatetimeIndex.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()]
        return out.sort_index()

    for candidate in ("datetime", "timestamp", "time", "date"):
        if candidate in out.columns:
            parsed = pd.to_datetime(out[candidate], utc=True, errors="coerce")
            out = out.drop(columns=[candidate])
            out.index = parsed
            out = out[~out.index.isna()]
            return out.sort_index()

    raise ValueError("DataFrame must have a DatetimeIndex or a datetime-like column.")


def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common broker/export column names to open/high/low/close/volume.
    """
    out = df.copy()
    column_map: dict[str, str] = {}
    for column in out.columns:
        name = str(column).strip().lower()
        if name == "open":
            column_map[column] = "open"
        elif name == "high":
            column_map[column] = "high"
        elif name == "low":
            column_map[column] = "low"
        elif name == "close":
            column_map[column] = "close"
        elif name in {"tick_volume", "volume", "vol"}:
            column_map[column] = "volume"

    out = out.rename(columns=column_map)
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise KeyError(f"Missing OHLC columns after standardisation: {missing}")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    ohlcv = out[["open", "high", "low", "close", "volume"]].copy()
    ohlcv = ohlcv.apply(pd.to_numeric, errors="coerce")
    return ohlcv.sort_index()


def resample_ohlcv(df: pd.DataFrame, target_tf: str = "1h") -> pd.DataFrame:
    """
    Resample a lower-timeframe OHLCV series to ``target_tf``.

    Rules:
    - open: first
    - high: max
    - low: min
    - close: last
    - volume: sum
    """
    indexed = _ensure_datetime_index(df)
    ohlcv = _standardize_ohlcv_columns(indexed)
    if ohlcv.empty:
        raise ValueError("No OHLCV rows available after standardisation.")

    resampled = ohlcv.resample(target_tf).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open"]).sort_index()
