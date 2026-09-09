#!/usr/bin/env python3
"""
gate0_reconcile.py — GATE 0: reproduce, reconcile, and leakage-check the frozen
H1 regime signal (ablation variant (a)) on EURUSD.

Three deliverables:
  1. Window A / Window B reproduction under the current (leaky) alignment, full
     stats + bootstrap CIs.
  2. +0.064R vs +0.0965R cost reconciliation (old 0.6-pip/$7 vs Cycle-4
     0.4-pip/$4), holding the 1,128 entries fixed.
  3. Leakage headline: honest alignment (last H1 bar CLOSED by the entry bar)
     vs the current forming-bar alignment, on all four available FX pairs.

    *** RESEARCH ONLY — no broker connectivity. ***
Leakage method: the current signal aligns H1 regime with merge_asof(backward)
on the H1 LABEL (open time); an M15 bar at :00/:15/:30 is matched to its own
still-forming H1 bar, whose close is future information. The honest alignment
matches on the H1 CLOSE time (label+1h), i.e. the most recent fully closed bar.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

_FORBIDDEN = ('MetaTrader5', 'mt5', 'ctrader_open_api', 'ib_insync', 'ccxt')
for _m in _FORBIDDEN:
    if _m in sys.modules:
        raise SystemExit(f"REFUSING TO RUN: broker module '{_m}' loaded.")

import backtest                    # noqa: E402
import config                      # noqa: E402
import strategy as S               # noqa: E402
import research_ablation as RA     # noqa: E402

FX = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
WINDOWS = {'A 2015-01→2025-03': ('2015-01-01', '2025-03-20'),
           'B 2007-01→2025-03': ('2007-01-01', '2025-03-20')}


def load(sym, lo, hi):
    d15 = backtest.get_ohlcv_from_csv(sym, 'M15')
    dh1 = backtest.get_ohlcv_from_csv(sym, 'H1')
    a, b = pd.Timestamp(lo, tz='UTC'), pd.Timestamp(hi, tz='UTC')
    return d15[(d15.index >= a) & (d15.index < b)], dh1[(dh1.index >= a) & (dh1.index < b)]


def boot(r, n=20000, seed=42, block=None):
    rng = np.random.default_rng(seed)
    m = len(r)
    if block:
        nb = int(np.ceil(m / block))
        st = rng.integers(0, m, size=(n, nb))
        idx = (st[:, :, None] + np.arange(block)[None, None, :]) % m
        means = r[idx.reshape(n, -1)[:, :m]].mean(1)
    else:
        means = r[rng.integers(0, m, size=(n, m))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def fullstats(r, times, label):
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    pf = g / l if l > 0 else float('inf')
    eq = np.cumprod(1 + r * 0 + r)  # R-space equity (additive) below
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0], cum]))
    ddR = (peak[1:] - cum).max()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    yrs = (times.max() - times.min()).days / 365.25
    ci = boot(r); cib = boot(r, block=10)
    print(f'{label}')
    print(f'  n={len(r)}  {len(r)/yrs:.0f}/yr  E[R]={r.mean():+.4f}  med={np.median(r):+.4f}  '
          f'sd={r.std(ddof=1):.3f}  WR={(r>0).mean()*100:.1f}%  PF={pf:.3f}')
    print(f'  maxDD={ddR:.2f}R  longest-losing-streak={best}  '
          f'iid95%[{ci[0]:+.4f},{ci[1]:+.4f}]  block95%[{cib[0]:+.4f},{cib[1]:+.4f}]')
    return r.mean()


def by_year(r, times):
    df = pd.DataFrame({'r': r, 'y': [t.year for t in times]})
    out = df.groupby('y')['r'].agg(['count', 'mean'])
    pos = (out['mean'] > 0).sum()
    print('  by year: ' + '  '.join(f'{y}:{m:+.3f}(n{int(c)})'
          for y, (c, m) in out.iterrows()))
    print(f'  years positive: {pos}/{len(out)}')


def leaky_R(sym, lo, hi):
    """Variant (a), current cost model, current (leaky) alignment."""
    d15, dh1 = load(sym, lo, hi)
    d = RA.run_ablation(d15, dh1, symbol=sym)['a_regime_only']
    return d['r'].to_numpy(float), d['entry_time'], d['direction'].to_numpy(int)


def honest_R(sym, lo, hi):
    """Variant (a) with the leak removed: regime = most recent H1 bar CLOSED by
    the entry bar's timestamp."""
    d15, dh1 = load(sym, lo, hi)
    eng = RA._prepared_engine(d15, dh1, symbol=sym)
    frame = S.build_signal_frame(eng.df, eng.df_h1, eng.params)
    eng._atrs = frame['atr_14'].to_numpy(float)
    index = eng.df.index
    reg = S.h1_regime(dh1)
    close_frame = pd.DataFrame({'regime': reg.to_numpy()},
                               index=reg.index + pd.Timedelta(hours=1))
    aligned = pd.merge_asof(
        pd.DataFrame(index=index).reset_index(names='t'),
        close_frame.reset_index(names='t').sort_values('t'),
        on='t', direction='backward').set_index('t')
    regime_avail = aligned['regime'].fillna(0).astype(int).to_numpy()
    session_ok = np.array([S.entry_session_ok(ts) for ts in index])
    opens = eng._opens
    entries, last_day = [], None
    for i in range(1, len(index)):
        day = index[i].date()
        if day == last_day or not session_ok[i] or regime_avail[i] == 0:
            continue
        sig = int(regime_avail[i])
        entries.append((i, sig, opens[i] + sig * eng._entry_cost(opens[i]), None))
        last_day = day
    d = RA._simulate(eng, entries)
    return d['r'].to_numpy(float), d['entry_time']


def main():
    lo, hi = WINDOWS['A 2015-01→2025-03']
    print('#' * 90)
    print('# DELIVERABLE 1 — reproduction under the CURRENT (leaky) alignment')
    print('#' * 90)
    for wl, (lo_, hi_) in WINDOWS.items():
        r, t, dirn = leaky_R('EURUSD', lo_, hi_)
        fullstats(r, t, f'EURUSD  Window {wl}  [current cost, LEAKY align]')
        by_year(r, t)
        lr, sr = r[dirn == 1], r[dirn == -1]
        print(f'  long : n={len(lr)}  E[R]={lr.mean():+.4f}  PF={_pf(lr):.3f}    '
              f'short: n={len(sr)}  E[R]={sr.mean():+.4f}  PF={_pf(sr):.3f}')
        cost_drag = (config.BACKTEST_COMMISSION_PER_LOT_RT)  # informational
        print()

    print('#' * 90)
    print('# DELIVERABLE 2 — +0.064R vs +0.0965R cost reconciliation (Window A, leaky)')
    print('#' * 90)
    combos = {'Cycle-4 verified (0.4pip,$4) -> expect +0.0965': (0.4, 4.0),
              'pre-Cycle-4 (0.6pip,$7)      -> expect +0.064':  (0.6, 7.0),
              'cross (0.6pip,$4)': (0.6, 4.0),
              'cross (0.4pip,$7)': (0.4, 7.0)}
    saved_sf = config.BACKTEST_SPREAD_FLOOR_BY_SYMBOL['EURUSD']
    saved_c = config.BACKTEST_COMMISSION_PER_LOT_RT
    for lbl, (sf, comm) in combos.items():
        config.BACKTEST_SPREAD_FLOOR_BY_SYMBOL['EURUSD'] = sf
        config.BACKTEST_COMMISSION_PER_LOT_RT = comm
        r, t, _ = leaky_R('EURUSD', lo, hi)
        print(f'  {lbl:52s}: n={len(r)}  E[R]={r.mean():+.4f}  PF={_pf(r):.3f}')
    config.BACKTEST_SPREAD_FLOOR_BY_SYMBOL['EURUSD'] = saved_sf
    config.BACKTEST_COMMISSION_PER_LOT_RT = saved_c

    print()
    print('#' * 90)
    print('# DELIVERABLE 3 — LEAKAGE HEADLINE: honest vs leaky, all four FX pairs (Window A)')
    print('#' * 90)
    print(f'{"pair":8s} {"leaky n":>8} {"leaky E[R]":>11} {"leaky PF":>9}   '
          f'{"honest n":>9} {"honest E[R]":>12} {"honest PF":>10}   {"ΔE[R]":>8}')
    for sym in FX:
        rl, tl, _ = leaky_R(sym, lo, hi)
        rh, th = honest_R(sym, lo, hi)
        cih = boot(rh)
        print(f'{sym:8s} {len(rl):>8} {rl.mean():>+11.4f} {_pf(rl):>9.3f}   '
              f'{len(rh):>9} {rh.mean():>+12.4f} {_pf(rh):>10.3f}   {rh.mean()-rl.mean():>+8.4f}'
              f'   honest iid95%[{cih[0]:+.4f},{cih[1]:+.4f}]')


def _pf(r):
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return g / l if l > 0 else float('inf')


if __name__ == '__main__':
    main()
