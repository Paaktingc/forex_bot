"""
Single-entry, single-stop, single-target/time-exit backtest for the intraday
continuation signal. Returns one after-cost R per trade. No grid, no averaging,
no trailing stop, no pyramiding.

Causality: entry is the OPEN of the bar AFTER the expansion bar; the stop
distance uses the ATR reference known at the signal bar (through t-1). Exits are
walked forward bar by bar; if stop and target are both touched in the same bar,
the stop is assumed first (conservative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from symbol_specs import get_symbol_spec

from . import config as C
from . import signal as sig


def _costs(pair: str, cost_mult: float = 1.0):
    spec = get_symbol_spec(pair)
    pip = spec.pip_size
    spread = C.SPREAD_FLOOR_PIPS[pair] * pip * cost_mult
    slip_entry = C.SLIPPAGE_ENTRY_PIPS * pip * cost_mult
    slip_stop = C.SLIPPAGE_STOP_PIPS * pip * cost_mult
    commission = C.COMMISSION_PER_LOT_RT * cost_mult
    return spec, pip, spread, slip_entry, slip_stop, commission


def run(df: pd.DataFrame, pair: str, *, tf: str = C.PRIMARY_TF,
        expansion_mult: float = C.EXPANSION_MULT, stop_atr: float = C.STOP_ATR,
        target_r: float = C.TARGET_R, time_exit_bars: int = C.TIME_EXIT_BARS,
        entry_delay: int = 1, cost_mult: float = 1.0,
        signals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Backtest the frozen signal on `df`. `entry_delay` bars after the signal
    bar (default 1 = next-bar open). Returns a per-trade DataFrame with after-
    and before-cost R.
    """
    spec, pip, spread, slip_entry, slip_stop, commission = _costs(pair, cost_mult)
    pip_value = spec.pip_value_per_standard_lot

    if signals is None:
        signals = sig.accepted_signals(df, tf=tf, expansion_mult=expansion_mult,
                                       stop_atr=stop_atr)

    idx = df.index
    pos = {ts: i for i, ts in enumerate(idx)}
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    tf_delta = pd.Timedelta(minutes=C.TF_MINUTES[tf])

    rows = []
    for ts, srow in signals.iterrows():
        i = pos[ts]
        j = i + entry_delay                     # entry bar
        if j >= len(idx):
            continue
        # reject if the entry bar is not contiguous (missing next bar)
        if (idx[j] - idx[i]) != tf_delta * entry_delay:
            continue
        d = int(srow["direction"])
        stop_dist = float(srow["stop_dist"])    # 1.0 * ATR_ref (price), causal
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            continue

        entry_gross = o[j]
        entry = entry_gross + d * (spread + slip_entry)   # pay spread+slip
        stop = entry - d * stop_dist
        target = entry + d * target_r * stop_dist
        commission_r = commission / ((stop_dist / pip) * pip_value)

        exit_reason, exit_price = None, None
        last = min(j + time_exit_bars - 1, len(idx) - 1)
        for k in range(j, last + 1):
            hit_stop = (l[k] <= stop) if d == 1 else (h[k] >= stop)
            hit_tgt = (h[k] >= target) if d == 1 else (l[k] <= target)
            if hit_stop and hit_tgt:
                exit_reason, exit_price = "stop_first", stop - d * slip_stop
                break
            if hit_stop:
                exit_reason, exit_price = "stop", stop - d * slip_stop
                break
            if hit_tgt:
                exit_reason, exit_price = "target", target
                break
        if exit_reason is None:
            exit_reason, exit_price = "time", c[last]

        r_gross = (exit_price - entry) * d / stop_dist
        r_net = r_gross - commission_r
        # before-cost variant: no spread/slip/commission
        r_beforecost = ((exit_price - entry_gross) * d
                        + (d * slip_stop if exit_reason in ("stop", "stop_first") else 0.0)) / stop_dist
        rows.append({
            "signal_time": ts, "entry_time": idx[j], "direction": d,
            "r": r_net, "r_gross": r_gross, "r_beforecost": r_beforecost,
            "exit_reason": exit_reason, "atr_ref": float(srow["atr_ref"]),
            "entry_hour": idx[j].tz_convert(C.SESSION_TZ).hour,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["vol_quartile"] = pd.qcut(out["atr_ref"], 4, labels=[1, 2, 3, 4],
                                      duplicates="drop")
    return out
