#!/usr/bin/env python3
"""
gate_a_per_instrument.py — GATE A of the pooled multi-instrument study.

Question: does ablation variant (a) (regime-only entry, first valid in-session
bar per day) replicate a positive expectancy on instruments OTHER than EURUSD?
Pooling only helps if the added instruments carry edge, not just noise.

Kill criterion: fewer than 3 instruments have a bootstrap CI on E[R] whose
lower bound is above 0.

    *** THIS CODE MUST NEVER TRADE. ***
Research-only. Reads CSV, writes stdout/CSV. No broker connectivity.

Leakage: the R series comes from research_ablation.run_ablation, which walks
bars forward one at a time — every indicator/threshold is trailing, SL/TP come
from the frozen production clamp at the entry bar, exits walk forward tick-order
(SL before TP). The design window ends before the 2025-03-20 lockbox, which is
not read here. Bootstrap resamples the realised R vector only; it introduces no
future information.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

_FORBIDDEN = ('MetaTrader5', 'mt5', 'ctrader_open_api', 'ib_insync', 'ccxt')
for _m in _FORBIDDEN:
    if _m in sys.modules:
        raise SystemExit(f"REFUSING TO RUN: broker module '{_m}' is loaded. "
                         "Research-only code; must not touch a live account.")

import backtest                    # noqa: E402
import research_ablation as RA     # noqa: E402

SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'XAUUSD', 'GRXEUR']
DESIGN_LO = pd.Timestamp('2015-01-01', tz='UTC')
LOCKBOX = pd.Timestamp('2025-03-20', tz='UTC')
SUB_2018 = pd.Timestamp('2018-01-01', tz='UTC')
N_BOOT = 20_000
SEED = 42
BLOCK = 10


def boot_ci_iid(r: np.ndarray, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(r)
    idx = rng.integers(0, m, size=(n, m))
    means = r[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), (means <= 0).mean()


def boot_ci_block(r: np.ndarray, block=BLOCK, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    m = len(r)
    b = max(1, min(block, m))
    nb = int(np.ceil(m / b))
    starts = rng.integers(0, m, size=(n, nb))
    off = np.arange(b)
    idx = (starts[:, :, None] + off[None, None, :]) % m
    means = r[idx.reshape(n, -1)[:, :m]].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def stats(r: np.ndarray) -> dict:
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    pf = g / l if l > 0 else float('inf')
    return dict(n=len(r), pf=pf, er=r.mean(), wr=(r > 0).mean() * 100)


def run_symbol(sym: str) -> dict:
    df15 = backtest.get_ohlcv_from_csv(sym, 'M15')
    dfh1 = backtest.get_ohlcv_from_csv(sym, 'H1')
    # Match STEP 1 / feasibility_envelope windowing exactly (same bounds on M15
    # and H1, no extra warmup) so EURUSD reproduces to the trade.
    m = (df15.index >= DESIGN_LO) & (df15.index < LOCKBOX)
    mh = (dfh1.index >= DESIGN_LO) & (dfh1.index < LOCKBOX)
    res = RA.run_ablation(df15[m], dfh1[mh], symbol=sym)
    d = res['a_regime_only']
    r = d['r'].to_numpy(float)
    t = d['entry_time']
    yrs = (t.max() - t.min()).days / 365.25
    s = stats(r)
    ci, p0 = boot_ci_iid(r)
    cib = boot_ci_block(r)
    # 2018+ subperiod
    sub = r[np.array([ts >= SUB_2018 for ts in t])]
    s_sub = stats(sub) if len(sub) else None
    ci_sub = boot_ci_iid(sub)[0] if len(sub) > 1 else (np.nan, np.nan)
    return dict(sym=sym, r=r, t=t, tpy=len(r) / yrs, **s,
                ci_lo=ci[0], ci_hi=ci[1], p_le0=p0,
                cib_lo=cib[0], cib_hi=cib[1],
                sub_n=(s_sub['n'] if s_sub else 0),
                sub_er=(s_sub['er'] if s_sub else np.nan),
                sub_ci_lo=ci_sub[0], sub_ci_hi=ci_sub[1])


def main():
    print('=' * 100)
    print('GATE A — ablation (a) per instrument, design window 2015-01-01 .. 2025-03-19')
    print('=' * 100)
    print('Leakage: trailing indicators, frozen entry-bar clamp, tick-order exits, '
          'lockbox untouched.\n')

    rows = []
    passes = 0
    for sym in SYMBOLS:
        d = run_symbol(sym)
        rows.append(d)
        lb_pos = d['ci_lo'] > 0 and d['cib_lo'] > 0
        passes += int(lb_pos)
        print(f"{d['sym']}  n={d['n']:>5}  {d['tpy']:>5.1f}/yr  PF={d['pf']:.3f}  "
              f"E[R]={d['er']:+.4f}  WR={d['wr']:.1f}%")
        print(f"    iid  95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  P(E[R]<=0)={d['p_le0']:.4f}")
        print(f"    block95% CI [{d['cib_lo']:+.4f}, {d['cib_hi']:+.4f}]   "
              f"{'CI_lo>0 ✅' if lb_pos else 'CI contains 0 ❌'}")
        print(f"    2018+  n={d['sub_n']:>5}  E[R]={d['sub_er']:+.4f}  "
              f"CI [{d['sub_ci_lo']:+.4f}, {d['sub_ci_hi']:+.4f}]\n")

    print('-' * 100)
    print(f"Instruments with both bootstrap CI lower bounds > 0: {passes}/6")
    print(f"KILL criterion (<3 pass): {'TRIGGERED — STOP' if passes < 3 else 'not triggered'}")

    out = pd.DataFrame([{k: v for k, v in d.items() if k not in ('r', 't')} for d in rows])
    out.to_csv('research/gate_a_per_instrument.csv', index=False)
    print('\nwritten: research/gate_a_per_instrument.csv')

    # EURUSD reproduction guardrail (must match STEP 1: n=1128, PF~1.185, E[R]~+0.0965)
    eu = next(d for d in rows if d['sym'] == 'EURUSD')
    print(f"\nGUARDRAIL EURUSD reproduction: n={eu['n']} (expect 1128)  "
          f"PF={eu['pf']:.3f} (expect ~1.185)  E[R]={eu['er']:+.4f} (expect ~+0.0965)")


if __name__ == '__main__':
    main()
