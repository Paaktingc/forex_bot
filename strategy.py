"""
strategy.py

Rules-based signal engine for the The5ers Bootcamp bot — the strategy IS the
rules; the ML model survives only as an optional veto (model.MetaVeto).

Regime (H1):
  long  iff EMA50 > EMA200 and close > EMA50
  short iff EMA50 < EMA200 and close < EMA50
  plus ADX(14) >= ADX_MIN gate (config.USE_ADX_GATE). No regime → no signal.

Entry (M15, regime direction only):
  pullback touches the M15 EMA20 (or the 38.2–61.8% retracement of the last
  H1 swing) within the lookback window, then an M15 close back in the trend
  direction with RSI(14) recrossing 50 → market order at next candle open.

All indicator math lives in features.py; this module only combines it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from features import (
    compute_entry_indicators_m15,
    compute_regime_indicators_h1,
    rolling_swing_levels,
)

logger = logging.getLogger(__name__)

# ~20 trading days of M15 bars, for the ATR-vs-median volatility filter
_BARS_PER_DAY_M15 = 96
ATR_MEDIAN_WINDOW_BARS = 20 * _BARS_PER_DAY_M15

RETRACEMENT_SHALLOW = 0.382
RETRACEMENT_DEEP = 0.618


@dataclass(frozen=True)
class StrategyParams:
    """Tunable rule parameters (perturbed in walk-forward robustness runs)."""

    ema_fast_h1: int = 50
    ema_slow_h1: int = 200
    ema_pullback: int = 20
    rsi_period: int = 14
    rsi_level: float = 50.0
    adx_min: float = field(default_factory=lambda: config.ADX_MIN)
    use_adx_gate: bool = field(default_factory=lambda: config.USE_ADX_GATE)
    pullback_lookback: int = 12          # M15 bars (~3 hours)
    swing_window_h1: int = 20            # H1 bars for the retracement swing
    sl_atr_mult: float = field(default_factory=lambda: config.SL_ATR_MULT)
    tp_r: float = field(default_factory=lambda: config.TP_R)


@dataclass(frozen=True)
class Candidate:
    """A rules-generated trade candidate (before filters / veto / risk)."""

    direction: int                       # 1 buy, -1 sell
    signal_time: pd.Timestamp            # close time of the trigger M15 bar
    swing_price: float                   # pullback extreme (SL anchor)
    atr: float                           # M15 ATR(14) at signal time
    atr_median: float | None = None      # trailing 20-day ATR median (vol filter)
    reason: str = "H1 regime + M15 pullback"


# ---------------------------------------------------------------------------
# Indicator assembly
# ---------------------------------------------------------------------------

def _ema_with_span(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def h1_regime(df_h1: pd.DataFrame, params: StrategyParams | None = None) -> pd.Series:
    """
    Per-H1-bar regime: +1 long, -1 short, 0 none. ADX gate applied when
    enabled. Uses only the completed bar's values.
    """
    params = params or StrategyParams()
    ind = compute_regime_indicators_h1(df_h1)
    close = ind["close"]
    ema_fast = _ema_with_span(close, params.ema_fast_h1)
    ema_slow = _ema_with_span(close, params.ema_slow_h1)

    regime = pd.Series(0, index=ind.index, dtype=int)
    regime[(ema_fast > ema_slow) & (close > ema_fast)] = 1
    regime[(ema_fast < ema_slow) & (close < ema_fast)] = -1

    if params.use_adx_gate:
        regime[~(ind["adx_14"] >= params.adx_min)] = 0

    regime[ema_slow.isna()] = 0
    return regime


def build_signal_frame(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    params: StrategyParams | None = None,
) -> pd.DataFrame:
    """
    Vectorized signal generation over a full M15 history (used by the
    backtester and, on the latest rows, by the live loop).

    Returns a frame indexed like df_m15 with columns:
      signal        +1/-1/0 at the trigger bar's close
      swing_price   pullback extreme for SL anchoring (NaN when no signal)
      atr_14        M15 ATR at the trigger bar
      atr_median    trailing 20-day median of atr_14 (volatility filter)
    """
    params = params or StrategyParams()

    m15 = compute_entry_indicators_m15(df_m15)
    close = m15["close"]
    ema_pull = _ema_with_span(close, params.ema_pullback)
    rsi = (
        m15["rsi_14"]
        if params.rsi_period == 14
        else _wilder_rsi(close, params.rsi_period)
    )

    # H1 regime and last-H1-swing retracement zone, aligned backward onto M15
    regime_h1 = h1_regime(df_h1, params)
    swings_h1 = rolling_swing_levels(df_h1, params.swing_window_h1)
    h1_frame = pd.DataFrame(
        {
            "regime": regime_h1,
            "swing_high_h1": swings_h1["swing_high"],
            "swing_low_h1": swings_h1["swing_low"],
        }
    )
    aligned = pd.merge_asof(
        pd.DataFrame(index=m15.index).reset_index(names="time"),
        h1_frame.reset_index(names="time").sort_values("time"),
        on="time",
        direction="backward",
    ).set_index("time")
    regime = aligned["regime"].fillna(0).astype(int)
    swing_high_h1 = aligned["swing_high_h1"]
    swing_low_h1 = aligned["swing_low_h1"]

    h1_range = (swing_high_h1 - swing_low_h1).replace(0, np.nan)
    # Long: pullback into 38.2–61.8% below the swing high; short mirrored
    long_zone_top = swing_high_h1 - RETRACEMENT_SHALLOW * h1_range
    long_zone_bottom = swing_high_h1 - RETRACEMENT_DEEP * h1_range
    short_zone_bottom = swing_low_h1 + RETRACEMENT_SHALLOW * h1_range
    short_zone_top = swing_low_h1 + RETRACEMENT_DEEP * h1_range

    low = m15["low"]
    high = m15["high"]

    long_touch = (low <= ema_pull) | ((low <= long_zone_top) & (high >= long_zone_bottom))
    short_touch = (high >= ema_pull) | ((high >= short_zone_bottom) & (low <= short_zone_top))

    lookback = params.pullback_lookback
    long_pullback = (
        long_touch.astype(float).rolling(lookback, min_periods=1).max()
        .shift(1).fillna(0.0).astype(bool)
    )
    short_pullback = (
        short_touch.astype(float).rolling(lookback, min_periods=1).max()
        .shift(1).fillna(0.0).astype(bool)
    )

    rsi_prev = rsi.shift(1)
    long_trigger = (
        (regime == 1)
        & long_pullback
        & (close > ema_pull)
        & (rsi > params.rsi_level)
        & (rsi_prev <= params.rsi_level)
    )
    short_trigger = (
        (regime == -1)
        & short_pullback
        & (close < ema_pull)
        & (rsi < params.rsi_level)
        & (rsi_prev >= params.rsi_level)
    )

    signal = pd.Series(0, index=m15.index, dtype=int)
    signal[long_trigger] = 1
    signal[short_trigger] = -1

    # SL anchor: the pullback extreme over the lookback window incl. trigger bar
    swing_low_m15 = low.rolling(lookback + 1, min_periods=1).min()
    swing_high_m15 = high.rolling(lookback + 1, min_periods=1).max()
    swing_price = pd.Series(np.nan, index=m15.index, dtype=float)
    swing_price[long_trigger] = swing_low_m15[long_trigger]
    swing_price[short_trigger] = swing_high_m15[short_trigger]

    atr = m15["atr_14"]
    return pd.DataFrame(
        {
            "signal": signal,
            "swing_price": swing_price,
            "atr_14": atr,
            "atr_median": atr.rolling(
                ATR_MEDIAN_WINDOW_BARS, min_periods=_BARS_PER_DAY_M15 * 5
            ).median(),
        },
        index=m15.index,
    )


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def generate_candidate(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    params: StrategyParams | None = None,
) -> Candidate | None:
    """
    Evaluates the LAST CLOSED M15 bar and returns a Candidate when the rules
    fire (execution happens at the next candle open), else None.
    """
    if df_m15 is None or df_m15.empty or df_h1 is None or df_h1.empty:
        return None

    frame = build_signal_frame(df_m15, df_h1, params)
    last = frame.iloc[-1]
    if int(last["signal"]) == 0:
        return None
    if not np.isfinite(last["swing_price"]) or not np.isfinite(last["atr_14"]):
        return None

    atr_median = float(last["atr_median"]) if np.isfinite(last["atr_median"]) else None
    return Candidate(
        direction=int(last["signal"]),
        signal_time=frame.index[-1],
        swing_price=float(last["swing_price"]),
        atr=float(last["atr_14"]),
        atr_median=atr_median,
    )


# ---------------------------------------------------------------------------
# Entry filters (spread / volatility / session)
# ---------------------------------------------------------------------------

def spread_ok(spread_pips: float) -> bool:
    """Skip entries when the live spread exceeds MAX_SPREAD_PIPS."""
    return 0 <= spread_pips <= config.MAX_SPREAD_PIPS


def volatility_ok(atr_pips: float, atr_median_pips: float | None) -> bool:
    """
    Skip when ATR(14, M15) < ATR_MIN_PIPS (dead market) or when ATR exceeds
    ATR_MAX_MEDIAN_MULT × its 20-day median (chaotic market).
    """
    if not np.isfinite(atr_pips) or atr_pips < config.ATR_MIN_PIPS:
        return False
    if atr_median_pips is not None and np.isfinite(atr_median_pips) and atr_median_pips > 0:
        if atr_pips > config.ATR_MAX_MEDIAN_MULT * atr_median_pips:
            return False
    return True


def entry_session_ok(when: datetime | pd.Timestamp) -> bool:
    """
    Entries only 08:00–17:00 Europe/London, Monday–Friday (DST-aware via
    zoneinfo). Additionally blocked:
      - Friday after FRIDAY_CUTOFF_LONDON (15:00 London)
      - the first SUNDAY_OPEN_BLOCK_HOURS after the Sunday market open
        (Sunday is fully outside the London window anyway)
    The 21:45–00:15 server-time window is enforced by the risk manager.
    """
    if isinstance(when, pd.Timestamp):
        when = when.to_pydatetime()
    if when.tzinfo is None:
        raise ValueError("entry_session_ok requires a timezone-aware datetime")

    local = when.astimezone(ZoneInfo(config.LONDON_TZ))
    weekday = local.weekday()  # Mon=0 … Sun=6

    if weekday >= 5:  # Saturday/Sunday (covers the Sunday-open block)
        return False
    # Monday first hours can still be within N hours of the Sunday open
    sunday_open_local = (local - timedelta(days=weekday + 1)).replace(
        hour=22, minute=0, second=0, microsecond=0
    )
    if timedelta(0) <= local - sunday_open_local < timedelta(
        hours=config.SUNDAY_OPEN_BLOCK_HOURS
    ):
        return False

    if not (config.SESSION_START_LONDON <= local.hour < config.SESSION_END_LONDON):
        return False
    if weekday == 4 and local.hour >= config.FRIDAY_CUTOFF_LONDON:
        return False
    return True
