#!/usr/bin/env python3
"""
gate0_leakage_probe.py — is the H1->M15 regime alignment look-ahead free?

The H1 CSVs are LEFT-labeled: an H1 bar stamped o covers [o, o+1h) and does not
close until o+1h. build_signal_frame aligns regime with merge_asof(direction=
'backward'), so an M15 bar at time t is matched to the H1 bar with label <= t —
which, for t at :00/:15/:30, is the bar CONTAINING t (still forming, closes in
the future). This probe measures whether variant (a) entries depend on that
forming-bar value.

Leak-safe alignment: shift the H1 regime by one bar before the backward merge,
so an M15 bar at t sees the most recent H1 bar that has fully CLOSED by t.

    *** RESEARCH ONLY — no broker connectivity. ***
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
import strategy as S               # noqa: E402
import research_ablation as RA     # noqa: E402
from risk_manager import compute_sl_tp  # noqa: E402

LO = pd.Timestamp('2015-01-01', tz='UTC')
HI = pd.Timestamp('2025-03-20', tz='UTC')


def regime_leaky(df15, dfh1) -> np.ndarray:
    """Current alignment: merge_asof backward on the H1 LABEL (open time).
    M15 bar t is matched to the H1 bar CONTAINING t (may still be forming)."""
    frame = pd.DataFrame({'regime': S.h1_regime(dfh1)})
    aligned = pd.merge_asof(
        pd.DataFrame(index=df15.index).reset_index(names='time'),
        frame.reset_index(names='time').sort_values('time'),
        on='time', direction='backward',
    ).set_index('time')
    return aligned['regime'].fillna(0).astype(int).to_numpy()


def regime_avail(df15, dfh1) -> np.ndarray:
    """Honest alignment: regime of the most recent H1 bar CLOSED by time t.
    An H1 bar labelled o closes at o+1h, so we align on close time."""
    reg = S.h1_regime(dfh1)
    frame = pd.DataFrame({'regime': reg.to_numpy()},
                         index=reg.index + pd.Timedelta(hours=1))  # close time
    aligned = pd.merge_asof(
        pd.DataFrame(index=df15.index).reset_index(names='time'),
        frame.reset_index(names='time').sort_values('time'),
        on='time', direction='backward',
    ).set_index('time')
    return aligned['regime'].fillna(0).astype(int).to_numpy()


def variant_a_entries(eng, regime_at_entry, index, session_ok, opens):
    """Variant (a): first in-session bar per day whose regime (as known at that
    bar) is non-zero. `regime_at_entry[i]` is the regime available for an entry
    executed at open[i]."""
    entries, last_day = [], None
    for i in range(1, len(index)):
        day = index[i].date()
        if day == last_day or not session_ok[i] or regime_at_entry[i] == 0:
            continue
        sig = int(regime_at_entry[i])
        entries.append((i, sig, opens[i] + sig * eng._entry_cost(opens[i]), None))
        last_day = day
    return entries


def er(r):
    return r.mean() if len(r) else float('nan')


def main():
    df15 = backtest.get_ohlcv_from_csv('EURUSD', 'M15')
    dfh1 = backtest.get_ohlcv_from_csv('EURUSD', 'H1')
    m = (df15.index >= LO) & (df15.index < HI)
    mh = (dfh1.index >= LO) & (dfh1.index < HI)
    d15, dh1 = df15[m], dfh1[mh]

    eng = RA._prepared_engine(d15, dh1, symbol='EURUSD')
    frame = S.build_signal_frame(eng.df, eng.df_h1, eng.params)
    eng._atrs = frame['atr_14'].to_numpy(float)
    index = eng.df.index
    session_ok = np.array([S.entry_session_ok(ts) for ts in index])
    opens = eng._opens

    reg_leaky = regime_leaky(d15, dh1)
    reg_avail = regime_avail(d15, dh1)

    # sanity: leaky regime here must match build_signal_frame's regime column
    bf_reg = frame['regime'].to_numpy(int)
    assert np.array_equal(reg_leaky, bf_reg), "probe leaky regime != build_signal_frame"

    # regime AS USED AT ENTRY:
    #   current code enters at open[i] using regime[i-1]  (leaky, may be forming)
    #   honest version uses the regime of the last H1 bar CLOSED by open[i]
    leaky_at_entry = np.empty_like(reg_leaky)
    leaky_at_entry[0] = 0
    leaky_at_entry[1:] = reg_leaky[:-1]            # regime[i-1] used at bar i
    safe_at_entry = reg_avail                       # regime known at bar i

    ent_leaky = variant_a_entries(eng, leaky_at_entry, index, session_ok, opens)
    ent_safe = variant_a_entries(eng, safe_at_entry, index, session_ok, opens)

    r_leaky = RA._simulate(eng, ent_leaky)
    r_safe = RA._simulate(eng, ent_safe)

    # how many entry decisions use a still-forming H1 bar (true leak)?
    leak_bars = int(sum(1 for i in range(1, len(index))
                        if leaky_at_entry[i] != 0 and leaky_at_entry[i] != safe_at_entry[i]))
    n_leaky_decisions = int((leaky_at_entry != 0).sum())

    print('=== GATE 0 leakage probe: EURUSD variant (a), design window ===')
    print(f'leaky (current: regime[i-1] at open[i])   : n={len(r_leaky)}  '
          f'E[R]={er(r_leaky["r"]):+.4f}  PF={_pf(r_leaky["r"]):.3f}')
    print(f'honest (last H1 bar CLOSED by open[i])    : n={len(r_safe)}  '
          f'E[R]={er(r_safe["r"]):+.4f}  PF={_pf(r_safe["r"]):.3f}')
    print(f'bars where current regime != honest regime (forming-bar look-ahead): '
          f'{leak_bars} of {n_leaky_decisions} non-zero decision bars '
          f'({100*leak_bars/max(n_leaky_decisions,1):.1f}%)')

    leaky_map = {i: sig for (i, sig, *_ ) in ent_leaky}
    safe_map = {i: sig for (i, sig, *_ ) in ent_safe}
    same_idx = set(leaky_map) & set(safe_map)
    dir_changes = sum(1 for i in same_idx if leaky_map[i] != safe_map[i])
    print(f'entry-bar overlap leaky∩honest={len(same_idx)}  dir changes={dir_changes}  '
          f'only-leaky={len(set(leaky_map)-set(safe_map))}  '
          f'only-honest={len(set(safe_map)-set(leaky_map))}')

    # E[R] by subset (simulate each subset through the same walker)
    rl = r_leaky.copy(); rl['idx'] = [i for (i, *_ ) in ent_leaky][:len(rl)]
    # rebuild idx->r maps by re-simulating filtered entry lists
    def sub_er(entlist, keep):
        sub = [e for e in entlist if e[0] in keep]
        d = RA._simulate(eng, sub)
        return len(d), (d['r'].mean() if len(d) else float('nan'))
    only_leaky_idx = set(leaky_map) - set(safe_map)
    only_honest_idx = set(safe_map) - set(leaky_map)
    nlc, erlc = sub_er(ent_leaky, same_idx)
    nlo, erlo = sub_er(ent_leaky, only_leaky_idx)
    nho, erho = sub_er(ent_safe, only_honest_idx)
    print(f'  common-bar entries      : n={nlc}  E[R]={erlc:+.4f}')
    print(f'  leak-only entries (look-ahead-created): n={nlo}  E[R]={erlo:+.4f}')
    print(f'  honest-only entries     : n={nho}  E[R]={erho:+.4f}')

    # spot check: at a top-of-hour entry, leaky and honest MUST use the same H1 bar
    hits = 0
    for i in range(2, 400):
        if index[i].minute == 0:  # top of hour
            assert leaky_at_entry[i] == safe_at_entry[i], (
                f'top-of-hour mismatch at {index[i]}: leaky={leaky_at_entry[i]} '
                f'honest={safe_at_entry[i]}')
            hits += 1
    print(f'  spot-check: {hits} top-of-hour bars all agree leaky==honest (as required)')


def _pf(r):
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return g / l if l > 0 else float('inf')


if __name__ == '__main__':
    main()
