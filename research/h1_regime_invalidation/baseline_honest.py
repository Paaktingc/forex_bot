#!/usr/bin/env python3
"""
baseline_honest.py — the leak-free H1 regime baseline, produced by the FIXED
strategy.build_signal_frame (closed-bar H1 alignment).

For each FX pair it reports the full honest stats and the entry-set
decomposition versus the OLD leaky (label-aligned) implementation:
  (1) leak-only entries, (2) common entries, (3) honest-only entries,
and the expectancy of each group.

    *** RESEARCH ONLY — no broker connectivity. ***
This reads CSV and prints. It does NOT enable or trade the retired strategy.
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
    if len(r) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed); m = len(r)
    if block:
        nb = int(np.ceil(m / block)); st = rng.integers(0, m, size=(n, nb))
        idx = (st[:, :, None] + np.arange(block)[None, None, :]) % m
        means = r[idx.reshape(n, -1)[:, :m]].mean(1)
    else:
        means = r[rng.integers(0, m, size=(n, m))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def _pf(r):
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return g / l if l > 0 else float('inf')


def honest(sym, lo, hi):
    """Variant (a) via the FIXED build_signal_frame. Returns (df, engine)."""
    d15, dh1 = load(sym, lo, hi)
    return RA.run_ablation(d15, dh1, symbol=sym)['a_regime_only']


def leaky_entry_times(sym, lo, hi):
    """Reconstruct the OLD leaky entry set: regime aligned by H1 LABEL."""
    d15, dh1 = load(sym, lo, hi)
    eng = RA._prepared_engine(d15, dh1, symbol=sym)
    frame = S.build_signal_frame(eng.df, eng.df_h1, eng.params)  # fixed regime
    eng._atrs = frame['atr_14'].to_numpy(float)
    index = eng.df.index
    reg = S.h1_regime(dh1)
    lf = pd.DataFrame({'regime': reg})
    aligned = pd.merge_asof(pd.DataFrame(index=index).reset_index(names='t'),
                            lf.reset_index(names='t').sort_values('t'),
                            on='t', direction='backward').set_index('t')
    reg_leaky = aligned['regime'].fillna(0).astype(int).to_numpy()
    session_ok = np.array([S.entry_session_ok(ts) for ts in index])
    opens = eng._opens
    entries, last_day = [], None
    for i in range(1, len(index)):
        day = index[i].date()
        if day == last_day or not session_ok[i] or reg_leaky[i - 1] == 0:
            continue
        entries.append((i, int(reg_leaky[i - 1]),
                        opens[i] + int(reg_leaky[i - 1]) * eng._entry_cost(opens[i]), None))
        last_day = day
    d = RA._simulate(eng, entries)
    return d


def stats_line(r, times, label):
    g, l = r[r > 0].sum(), -r[r < 0].sum(); pf = g / l if l > 0 else float('inf')
    cum = np.cumsum(r); peak = np.maximum.accumulate(np.concatenate([[0], cum]))
    ddR = (peak[1:] - cum).max()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0; best = max(best, streak)
    yrs = (times.max() - times.min()).days / 365.25
    ci = boot(r); cib = boot(r, block=10)
    print(f'{label}')
    print(f'  n={len(r)}  {len(r)/yrs:.0f}/yr  E[R]={r.mean():+.4f}  med={np.median(r):+.4f}  '
          f'WR={(r>0).mean()*100:.1f}%  PF={pf:.3f}  maxDD={ddR:.1f}R  streak={best}')
    print(f'  iid95%[{ci[0]:+.4f},{ci[1]:+.4f}]  block95%[{cib[0]:+.4f},{cib[1]:+.4f}]')


def main():
    print('=' * 90)
    print('HONEST H1 REGIME BASELINE — fixed closed-bar alignment (variant a)')
    print('=' * 90)

    print('\n--- EURUSD, both windows ---')
    for wl, (lo, hi) in WINDOWS.items():
        d = honest('EURUSD', lo, hi)
        r = d['r'].to_numpy(float); dirn = d['direction'].to_numpy(int)
        stats_line(r, d['entry_time'], f'EURUSD Window {wl}')
        df = pd.DataFrame({'r': r, 'y': [t.year for t in d['entry_time']]})
        yr = df.groupby('y')['r'].mean()
        print('  by year: ' + ' '.join(f'{y}:{m:+.2f}' for y, m in yr.items())
              + f'   (pos {int((yr>0).sum())}/{len(yr)})')
        lr, sr = r[dirn == 1], r[dirn == -1]
        print(f'  long n={len(lr)} E[R]={lr.mean():+.4f} PF={_pf(lr):.3f}   '
              f'short n={len(sr)} E[R]={sr.mean():+.4f} PF={_pf(sr):.3f}')

    lo, hi = WINDOWS['A 2015-01→2025-03']
    print('\n--- Four-pair honest table + entry-set decomposition (Window A) ---')
    print(f'{"pair":8s} {"n":>5} {"E[R]":>9} {"PF":>7} {"iid95% CI":>22}  '
          f'{"removed/changed vs leaky":>26}')
    for sym in FX:
        dh = honest(sym, lo, hi)
        dl = leaky_entry_times(sym, lo, hi)
        rh = dh['r'].to_numpy(float)
        th = set(pd.to_datetime(dh['entry_time']))
        tl = set(pd.to_datetime(dl['entry_time']))
        common = th & tl; only_h = th - tl; only_l = tl - th
        ci = boot(rh)
        # subset expectancies
        hmap = {pd.Timestamp(t): r for t, r in zip(dh['entry_time'], rh)}
        lmap = {pd.Timestamp(t): r for t, r in zip(dl['entry_time'], dl['r'].to_numpy(float))}
        er_common = np.mean([hmap[t] for t in common]) if common else float('nan')
        er_only_h = np.mean([hmap[t] for t in only_h]) if only_h else float('nan')
        er_only_l = np.mean([lmap[t] for t in only_l]) if only_l else float('nan')
        print(f'{sym:8s} {len(rh):>5} {rh.mean():>+9.4f} {_pf(rh):>7.3f} '
              f'[{ci[0]:+.4f},{ci[1]:+.4f}]   leaky_n={len(dl)} '
              f'common={len(common)} leak_only={len(only_l)} honest_only={len(only_h)}')
        print(f'         decomposition E[R]: common={er_common:+.4f}  '
              f'leak_only(false edge)={er_only_l:+.4f}  honest_only={er_only_h:+.4f}')

    print('\nCost convention: EURUSD spread floor '
          f'{config.BACKTEST_SPREAD_FLOOR_BY_SYMBOL["EURUSD"]} pip, '
          f'commission ${config.BACKTEST_COMMISSION_PER_LOT_RT}/lot RT (current).')


if __name__ == '__main__':
    main()
