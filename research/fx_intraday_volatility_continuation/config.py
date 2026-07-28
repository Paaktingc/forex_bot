"""
Frozen configuration for the FX intraday volatility-expansion continuation
study. ALL definitions are declared here BEFORE any performance is examined and
must not be changed after seeing pair results (Gate 2 freeze).

RESEARCH ONLY. This package never imports MetaTrader5/broker SDKs, never places
orders, and never reuses the retired H1 regime strategy's parameters or code.
"""
from __future__ import annotations

import pandas as pd

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]

# --- chronological split (locked before any result) -------------------------
DISCOVERY = ("2007-01-01", "2018-12-31")
VALIDATION = ("2019-01-01", "2022-12-31")
LOCKED_OOS = ("2023-01-01", None)          # None = latest complete date
SPLITS = {"discovery": DISCOVERY, "validation": VALIDATION, "locked_oos": LOCKED_OOS}

# --- frozen signal specification (Gate 2) -----------------------------------
ATR_LEN = 20                 # ATR(20), trailing; reference is ATR through t-1
EXPANSION_MULT = 1.50        # true_range >= 1.5 * ATR_reference
CLOSE_LOCATION = 0.75        # (close-low)/(high-low) >= 0.75 for a bullish bar
BREAKOUT_LOOKBACK = 8        # close beyond high/low of previous 8 CLOSED bars
SESSION_LONDON = (8, 16)     # expansion bar CLOSE within [08:00, 16:00) London
SESSION_TZ = "Europe/London"
COOLDOWN_HOURS = 4           # <=1 entry per pair per rolling 4h (from signal ts)
STOP_ATR = 1.0               # initial stop = 1.0 * ATR_reference
TARGET_R = 1.5               # fixed target at 1.5R
TIME_EXIT_BARS = 16          # close at market after 16 bars if no stop/target
SAME_BAR_RULE = "stop_first"  # if stop & target both touched intrabar

PRIMARY_TF = "M15"
TF_MINUTES = {"M15": 15, "M30": 30}

# --- cost convention (current repo values; see config.py) -------------------
# Spread floors in pips per pair, commission USD/lot round-trip, entry/stop
# slippage in pips. Verdicts use AFTER-cost results.
SPREAD_FLOOR_PIPS = {"EURUSD": 0.4, "GBPUSD": 0.6, "AUDUSD": 0.6, "USDJPY": 0.5}
COMMISSION_PER_LOT_RT = 4.0
SLIPPAGE_ENTRY_PIPS = 0.3
SLIPPAGE_STOP_PIPS = 1.0

SEED = 20260729
N_BOOT = 10_000


def split_bounds(name: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    lo, hi = SPLITS[name]
    return (pd.Timestamp(lo, tz="UTC"),
            pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1) if hi else None)
