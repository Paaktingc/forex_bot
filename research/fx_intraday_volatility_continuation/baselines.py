"""
Baselines for Gate 3:
  1. matched random entry  (destroys the signal, matches count/session-hour/
     direction-frequency/vol-quartile/holding rules)
  2. shuffled signal labels (randomises expansion direction at the true signal
     bars, preserves timestamps/vol/management)
  3. unconditional session continuation (enter with the sign of ANY in-session
     M15 bar; same management + cooldown)

All baselines reuse the SAME trade management and cost model as the strategy.
A vectorised forward-outcome engine computes the after-cost R of entering at
each bar's next open, for both directions, once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from symbol_specs import get_symbol_spec

from . import config as C
from . import signal as sig
from .prepare_data import add_session_flag


def _cost_params(pair, cost_mult=1.0):
    spec = get_symbol_spec(pair)
    pip = spec.pip_size
    return (spec, pip, C.SPREAD_FLOOR_PIPS[pair] * pip * cost_mult,
            C.SLIPPAGE_ENTRY_PIPS * pip * cost_mult,
            C.SLIPPAGE_STOP_PIPS * pip * cost_mult,
            C.COMMISSION_PER_LOT_RT * cost_mult)


def forward_outcomes(df: pd.DataFrame, pair: str, direction: int,
                     stop_dist: np.ndarray, *, tf: str = C.PRIMARY_TF,
                     target_r: float = C.TARGET_R,
                     time_exit_bars: int = C.TIME_EXIT_BARS,
                     entry_delay: int = 1, cost_mult: float = 1.0) -> np.ndarray:
    """After-cost R of entering `direction` at bar (b+entry_delay) open with the
    given per-bar stop distance, for EVERY bar b. NaN where no valid entry.
    Vectorised over bars; same-bar stop assumed first."""
    spec, pip, spread, slip_entry, slip_stop, commission = _cost_params(pair, cost_mult)
    pip_value = spec.pip_value_per_standard_lot
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = df.index
    n = len(idx)
    tf_delta = pd.Timedelta(minutes=C.TF_MINUTES[tf])
    d = direction

    b = np.arange(n)
    j = b + entry_delay
    valid = (j < n) & np.isfinite(stop_dist) & (stop_dist > 0)
    valid[:] &= np.concatenate([
        (idx[entry_delay:] - idx[:-entry_delay] == tf_delta * entry_delay),
        np.zeros(entry_delay, bool)])
    R = np.full(n, np.nan)
    jj = j[valid]
    sd = stop_dist[valid]
    entry_gross = o[jj]
    entry = entry_gross + d * (spread + slip_entry)
    stop = entry - d * sd
    target = entry + d * target_r * sd
    commission_r = commission / ((sd / pip) * pip_value)

    m = len(jj)
    done = np.zeros(m, bool)
    exit_r = np.zeros(m)
    for k in range(time_exit_bars):
        kb = jj + k
        ok = kb < n
        hk, lk = np.where(ok, h[np.clip(kb, 0, n - 1)], np.nan), np.where(ok, l[np.clip(kb, 0, n - 1)], np.nan)
        if d == 1:
            hit_stop = lk <= stop
            hit_tgt = hk >= target
            sp = stop - slip_stop
        else:
            hit_stop = hk >= stop
            hit_tgt = lk <= target
            sp = stop + slip_stop
        newly_stop = ok & ~done & hit_stop            # stop-first on ties
        newly_tgt = ok & ~done & ~hit_stop & hit_tgt
        exit_r[newly_stop] = (sp[newly_stop] - entry[newly_stop]) * d / sd[newly_stop]
        exit_r[newly_tgt] = (target[newly_tgt] - entry[newly_tgt]) * d / sd[newly_tgt]
        done |= newly_stop | newly_tgt
    # time exit for survivors at close of last held bar
    last = np.clip(jj + time_exit_bars - 1, 0, n - 1)
    surv = ~done
    exit_r[surv] = (c[last[surv]] - entry[surv]) * d / sd[surv]
    R[np.where(valid)[0]] = exit_r - commission_r
    return R


def pool_frame(df: pd.DataFrame, pair: str, *, tf: str = C.PRIMARY_TF,
               cost_mult: float = 1.0) -> pd.DataFrame:
    """In-session candidate bars with precomputed r_long / r_short and context."""
    atr_ref = sig.atr_reference(df)
    stop_dist = (C.STOP_ATR * atr_ref).to_numpy(float)
    session = add_session_flag(df, tf).to_numpy()
    r_long = forward_outcomes(df, pair, 1, stop_dist, tf=tf, cost_mult=cost_mult)
    r_short = forward_outcomes(df, pair, -1, stop_dist, tf=tf, cost_mult=cost_mult)
    out = pd.DataFrame({
        "r_long": r_long, "r_short": r_short, "atr_ref": atr_ref.to_numpy(float),
        "hour": df.index.tz_convert(C.SESSION_TZ).hour,
        "year": df.index.year, "in_session": session,
    }, index=df.index)
    out["valid"] = out["in_session"] & np.isfinite(out["r_long"]) & np.isfinite(out["r_short"])
    return out


def matched_random(strat: pd.DataFrame, pool: pd.DataFrame, *,
                   n_resamples: int = C.N_BOOT, seed: int = C.SEED) -> np.ndarray:
    """Mean-R distribution of random entries matched on (year, hour, vol
    quartile) cells and the strategy's exact per-trade direction."""
    p = pool[pool["valid"]].copy()
    edges = np.quantile(strat["atr_ref"], [0, .25, .5, .75, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    p["vq"] = pd.cut(p["atr_ref"], np.unique(edges), labels=False, include_lowest=True)
    st = strat.copy()
    st["vq"] = pd.cut(st["atr_ref"], np.unique(edges), labels=False, include_lowest=True)

    # group pool rows by matching cell -> arrays of (r_long, r_short)
    groups = {}
    for (yr, hr, vq), g in p.groupby(["year", "hour", "vq"]):
        groups[(yr, hr, vq)] = (g["r_long"].to_numpy(), g["r_short"].to_numpy())
    by_hour = {}
    for (hr), g in p.groupby("hour"):
        by_hour[hr] = (g["r_long"].to_numpy(), g["r_short"].to_numpy())

    rng = np.random.default_rng(seed)
    years = st["year"].to_numpy() if "year" in st else np.array([t.year for t in st.index])
    hours = (st["entry_hour"] if "entry_hour" in st else st["hour"]).to_numpy()
    vqs = st["vq"].to_numpy()
    dirs = st["direction"].to_numpy()
    # trade-outer, resample-inner (vectorised): sum of per-trade draws / n_trades
    acc = np.zeros(n_resamples)
    for hr, vq, d, yr in zip(hours, vqs, dirs, years):
        cell = groups.get((yr, hr, vq)) or by_hour.get(hr)
        arr = cell[0] if d == 1 else cell[1]
        acc += arr[rng.integers(len(arr), size=n_resamples)]
    return acc / len(dirs)


def shuffled_labels(strat: pd.DataFrame, pool: pd.DataFrame, *,
                    n_resamples: int = C.N_BOOT, seed: int = C.SEED) -> np.ndarray:
    """Randomise long/short at the TRUE signal bars (preserve timestamps/vol/
    management). Uses precomputed r_long/r_short at those bars."""
    p = pool.reindex(strat.index)
    rl, rs = p["r_long"].to_numpy(), p["r_short"].to_numpy()
    ok = np.isfinite(rl) & np.isfinite(rs)
    rl, rs = rl[ok], rs[ok]
    rng = np.random.default_rng(seed + 7)
    flips = rng.integers(0, 2, size=(n_resamples, len(rl)))
    picked = np.where(flips == 1, rl[None, :], rs[None, :])
    return picked.mean(axis=1)


def unconditional_continuation(df: pd.DataFrame, pair: str, *,
                               tf: str = C.PRIMARY_TF, cost_mult: float = 1.0,
                               cooldown_hours: float = C.COOLDOWN_HOURS) -> np.ndarray:
    """Enter with the sign of ANY completed in-session bar (no expansion/
    breakout/close-location), same management + cooldown. Returns per-trade R."""
    atr_ref = sig.atr_reference(df)
    stop_dist = (C.STOP_ATR * atr_ref).to_numpy(float)
    session = add_session_flag(df, tf).to_numpy()
    body = (df["close"] - df["open"]).to_numpy()
    direction = np.sign(body).astype(int)
    cand = np.where(session & (direction != 0) & np.isfinite(stop_dist) & (stop_dist > 0))[0]
    # cooldown from signal timestamp
    cd = pd.Timedelta(hours=cooldown_hours)
    idx = df.index
    accepted, last = [], None
    for b in cand:
        ts = idx[b]
        if last is None or (ts - last) >= cd:
            accepted.append(b); last = ts
    rl = forward_outcomes(df, pair, 1, stop_dist, tf=tf, cost_mult=cost_mult)
    rs = forward_outcomes(df, pair, -1, stop_dist, tf=tf, cost_mult=cost_mult)
    out = [rl[b] if direction[b] == 1 else rs[b] for b in accepted]
    return np.array([x for x in out if np.isfinite(x)])
