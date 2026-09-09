#!/usr/bin/env python3
"""
feasibility_envelope.py — STEP 1 of the post-Gate-0 plan.

Question: given the ONLY surviving positive signal in this repo (ablation
variant (a), regime-only entry), does ANY risk-per-trade setting simultaneously
(i)  give a usable probability of reaching the programme profit target, and
(ii) hold P(maxDD > 5%) < 5%?

If no risk level clears both, the answer is NO-GO and no further engineering
changes it. This is deliberately run BEFORE building any strategy code.

    *** THIS CODE MUST NEVER TRADE. ***
Research-only. Reads CSV, writes CSV/stdout. No broker connectivity.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

_FORBIDDEN = ('MetaTrader5', 'mt5', 'ctrader_open_api', 'ib_insync', 'ccxt')
for _m in _FORBIDDEN:
    if _m in sys.modules:
        raise SystemExit(f"REFUSING TO RUN: broker module '{_m}' is loaded. "
                         "This is research-only code and must not touch a live account.")

import backtest                      # noqa: E402
import research_ablation as RA       # noqa: E402
from monte_carlo_dd import (         # noqa: E402
    block_bootstrap_paths, _step_outcomes_vectorized, DEFAULT_BLOCK_SIZE,
)

N_PATHS = 20_000
SEED = 42
RISKS = [0.0020, 0.0025, 0.0030, 0.0040, 0.0050, 0.0060, 0.0075, 0.0100]

# Programme geometry (config.py PROGRAMMES, verified 2026-07-21).
PROGRAMMES = {
    'Bootcamp  (3x +6%, -5% static)': dict(steps=(0.06, 0.06, 0.06), fail=-0.05, kill=-0.03),
    'High Stakes (10%+5%, -10%)':     dict(steps=(0.10, 0.05),       fail=-0.10, kill=-0.06),
}


def get_r_series(lo: str, hi: str) -> np.ndarray:
    """Ablation (a) per-trade R multiples, after costs, on [lo, hi)."""
    df15 = backtest.get_ohlcv_from_csv('EURUSD', 'M15')
    dfh1 = backtest.get_ohlcv_from_csv('EURUSD', 'H1')
    a, b = pd.Timestamp(lo, tz='UTC'), pd.Timestamp(hi, tz='UTC')
    res = RA.run_ablation(df15[(df15.index >= a) & (df15.index < b)],
                          dfh1[(dfh1.index >= a) & (dfh1.index < b)])
    d = res['a_regime_only']
    return d['r'].to_numpy(float), d['entry_time']


def describe(r: np.ndarray, times, label: str) -> float:
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    yrs = (times.max() - times.min()).days / 365.25
    tpy = len(r) / yrs
    print(f'{label}')
    print(f'  n={len(r)}  PF={g/l:.3f}  E[R]={r.mean():+.4f}  WR={(r>0).mean()*100:.1f}%  '
          f'sd(R)={r.std(ddof=1):.3f}')
    print(f'  span {times.min().date()} .. {times.max().date()}  = {yrs:.1f}y  '
          f'-> {tpy:.0f} trades/year')
    print(f'  worst single R = {r.min():.3f}   (values < -1.0 = slippage through the stop)')
    return tpy


def p_target_before_X(paths: np.ndarray, target: float, X: float) -> float:
    """P(equity reaches +target before it ever touches -X)."""
    eq = np.cumprod(1.0 + paths, axis=1)
    big = paths.shape[1] + 1
    fi = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), big)   # noqa: E731
    return float((fi(eq >= 1 + target) < fi(eq <= 1 - X)).mean())


def max_dd_dist(paths: np.ndarray, horizon: int) -> np.ndarray:
    """Max drawdown of each UNABSORBED path over `horizon` trades.

    No kill switch, no absorption: this measures how risky the return stream
    itself is, which is what P(maxDD > 5%) is asking. The absorbed-path version
    is reported separately as p_breach.
    """
    eq = np.cumprod(1.0 + paths[:, :horizon], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    return (peak - eq).max(axis=1) / peak.max(axis=1)


def main():
    print('=' * 100)
    print('STEP 1 — FEASIBILITY ENVELOPE for ablation (a), regime-only EURUSD entry')
    print('=' * 100)

    print('\nLeakage verification: the R series is produced by research_ablation.py, which '
          'walks\nbars forward one at a time; every indicator is trailing, the SL/TP come from '
          'the frozen\nproduction clamp computed at the entry bar, and the design window ends '
          'before the\n2025-03-20 lockbox. The lockbox is NOT touched here. Risk levels are a '
          'fixed a-priori\nsweep, not fitted to the outcome.\n')

    r_dw, t_dw = get_r_series('2015-01-01', '2025-03-20')
    tpy = describe(r_dw, t_dw, 'DESIGN WINDOW 2015-01 -> 2025-03-19 (protocol window)')
    r_full, t_full = get_r_series('2007-01-01', '2025-03-20')
    describe(r_full, t_full, '\nFULL PRE-LOCKBOX 2007-01 -> 2025-03-19 (sensitivity only)')

    horizon = int(round(tpy))     # one year of trading
    print(f'\nMonte Carlo: {N_PATHS:,} block-bootstrap paths, block={DEFAULT_BLOCK_SIZE}, '
          f'seed={SEED}.\nmaxDD horizon = {horizon} trades = 1 year.\n')

    rows = []
    for label, geo in PROGRAMMES.items():
        step1 = geo['steps'][0]
        print('=' * 100)
        print(f'{label}   step-1 target +{step1*100:.0f}%, official max loss '
              f'{geo["fail"]*100:.0f}%, kill {geo["kill"]*100:.0f}%')
        print('=' * 100)
        hdr = (f'{"risk":>6} {"P(pass s1)":>11} {"P(kill)":>8} {"P(breach)":>10} '
               f'{"med trades":>11} {"~months":>8} {"P(mDD>5%)":>10} {"P(all steps)":>13}')
        print(hdr)
        for f in RISKS:
            pct = r_dw * f
            paths = block_bootstrap_paths(pct, n_paths=N_PATHS,
                                          block_size=DEFAULT_BLOCK_SIZE, seed=SEED)
            oc, tr = _step_outcomes_vectorized(paths, target=step1,
                                               fail=geo['fail'], kill=geo['kill'])
            p_pass = (oc == 0).mean()
            p_kill = (oc == 1).mean()
            p_breach = (oc == 2).mean()
            med = np.median(tr[oc == 0]) if (oc == 0).any() else np.nan
            months = med / tpy * 12 if np.isfinite(med) else np.nan
            pdd5 = (max_dd_dist(paths, horizon) > 0.05).mean()
            # naive independent chaining of the steps
            p_all = 1.0
            for st in geo['steps']:
                o2, _ = _step_outcomes_vectorized(paths, target=st,
                                                  fail=geo['fail'], kill=geo['kill'])
                p_all *= (o2 == 0).mean()
            print(f'{f*100:>5.2f}% {p_pass:>11.3f} {p_kill:>8.3f} {p_breach:>10.4f} '
                  f'{med:>11.0f} {months:>8.1f} {pdd5:>10.4f} {p_all:>13.3f}')
            rows.append(dict(programme=label, risk_pct=f * 100, p_pass_step1=round(p_pass, 4),
                             p_kill=round(p_kill, 4), p_breach=round(p_breach, 4),
                             median_trades_to_pass=float(med), months_to_pass=round(months, 1),
                             p_maxdd_gt_5pct=round(pdd5, 4), p_all_steps=round(p_all, 4)))
        print()

    # ── P(reach +6% before breaching X) — the Gate-4 table, Bootcamp geometry ──
    print('=' * 100)
    print('P(reach +6% before EVER touching -X)  — Bootcamp step 1, no kill switch')
    print('=' * 100)
    print(f'{"risk":>6}' + ''.join(f'{f"X={x}%":>10}' for x in [1, 2, 3, 4, 5]))
    for f in RISKS:
        paths = block_bootstrap_paths(r_dw * f, n_paths=N_PATHS,
                                      block_size=DEFAULT_BLOCK_SIZE, seed=SEED)
        print(f'{f*100:>5.2f}%' + ''.join(
            f'{p_target_before_X(paths, 0.06, x/100):>10.3f}' for x in [1, 2, 3, 4, 5]))

    pd.DataFrame(rows).to_csv('research/feasibility_envelope.csv', index=False)
    print('\nwritten: research/feasibility_envelope.csv')


if __name__ == '__main__':
    main()
