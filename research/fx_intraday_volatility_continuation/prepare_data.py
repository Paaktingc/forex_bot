"""
Data loading, split enforcement and the locked-out-of-sample guard for the
intraday continuation study.

Causality contract:
- M15/M30 bars are LEFT-labelled (index = bar OPEN, UTC). A bar labelled t
  covers [t, t+tf) and CLOSES at t+tf. Decisions are taken at bar CLOSE.
- The locked OOS period (2023-01-01 .. latest) is unreachable during discovery
  and validation: load_split() raises unless unlock_oos=True is passed
  explicitly (only the final locked test may do so).

RESEARCH ONLY. No broker imports.
"""
from __future__ import annotations

import sys

import pandas as pd

import data_feed  # CSV reader only
from resampler import resample_ohlcv

from . import config as C

_FORBIDDEN = ("MetaTrader5", "mt5", "ctrader_open_api", "ib_insync", "ccxt")


def assert_research_only() -> None:
    """Refuse to RUN a research entry point if any broker SDK is loaded in the
    process. Called by the runners (not at import, so pytest can import the
    package alongside broker-touching tests)."""
    for _m in _FORBIDDEN:
        if _m in sys.modules:
            raise SystemExit(
                f"REFUSING TO RUN: broker module '{_m}' loaded. This is "
                "research-only code and must not touch a live account."
            )


class LockboxViolation(RuntimeError):
    """Raised on any attempt to read the locked OOS period without unlock."""


def load_m15(pair: str) -> pd.DataFrame:
    """Full clean M15 history (UTC, left-labelled)."""
    df = data_feed.get_ohlcv_from_csv(pair, "M15")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df[["open", "high", "low", "close"]].sort_index()


def to_m30(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Causal M15->M30 resample (left-labelled, closed-left), for the Gate-4
    robustness comparison only. No M30 CSVs exist in the repo."""
    return resample_ohlcv(df_m15, "30min")[["open", "high", "low", "close"]]


def load_split(pair: str, split: str, *, tf: str = "M15",
               unlock_oos: bool = False) -> pd.DataFrame:
    """Return the bars for `pair` within `split`.

    The locked OOS split raises LockboxViolation unless unlock_oos=True.
    """
    if split == "locked_oos" and not unlock_oos:
        raise LockboxViolation(
            "locked_oos is sealed during discovery/validation. Pass "
            "unlock_oos=True only in the final locked test."
        )
    df = load_m15(pair)
    if tf == "M30":
        df = to_m30(df)
    elif tf != "M15":
        raise ValueError(f"unsupported tf {tf!r}")
    lo, hi = C.split_bounds(split)
    m = df.index >= lo
    if hi is not None:
        m &= df.index < hi
    return df[m]


def add_session_flag(df: pd.DataFrame, tf: str = "M15") -> pd.Series:
    """True where the bar's CLOSE time (label + tf) is within the London
    session window [08:00, 16:00). Timezone-aware -> DST handled automatically."""
    close_utc = df.index + pd.Timedelta(minutes=C.TF_MINUTES[tf])
    local = close_utc.tz_convert(C.SESSION_TZ)
    lo, hi = C.SESSION_LONDON
    return pd.Series((local.hour >= lo) & (local.hour < hi), index=df.index)
