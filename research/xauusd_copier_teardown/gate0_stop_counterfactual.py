#!/usr/bin/env python3
"""
gate0_stop_counterfactual.py — GATE 0 of the XAUUSD basket investigation.

Question: was the original copier's entire return produced by the deferred-loss
mechanism (no stop-loss), or does it survive a hard basket stop-loss?

Method
------
Re-simulate the 442 tick-covered baskets with a hard stop on the basket's
AGGREGATE FLOATING P/L. Everything else is held identical: same entries, same
lot ladder, same exit at weighted-average entry + ~2.3 USD/oz.

The counterfactual is exact given `min_float`, which `pipeline/tick_analysis.py`
computed from 6,281,860 Dukascopy ticks as the running minimum of the basket's
aggregate floating P/L, counting ONLY positions already opened at each second
(`live = (tsec >= opens)`). Therefore:

    basket is stopped  <=>  min_float <= -stop_usc

because the floating-P/L path up to the first crossing of -stop_usc is
identical in the counterfactual (all entries before that moment are unchanged),
and the stop simply truncates the path there. Entries that would have followed
the stop never happen, which does not alter the realised loss (= -stop_usc).

    *** THIS CODE MUST NEVER TRADE. ***
Research-only. No broker connectivity. Reads CSV, writes CSV/stdout.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

# ───────────────────────── SAFETY GUARD (same pattern as research_backtest.py)
_FORBIDDEN = ('MetaTrader5', 'mt5', 'ctrader_open_api', 'ib_insync', 'ccxt')
for _m in _FORBIDDEN:
    if _m in sys.modules:
        raise SystemExit(f"REFUSING TO RUN: broker module '{_m}' is loaded. "
                         "This is research-only code and must not touch a live account.")

DATA = 'data/baskets_with_mae.csv'
BALANCE = 101_942.72          # USC, ledger balance at 2026-06-30 EOD (raw_summary Sheet14)
STOPS = [0.005, 0.010, 0.020, 0.030]

# Cost of closing at the stop, in USC per lot-oz of exposure. The floating P/L
# in tick_analysis.py is already marked worst-side (bid for longs, ask for
# shorts), i.e. spread is included. We add a slippage allowance on top.
SLIP_USD_PER_OZ = 0.10        # USD/oz adverse slippage on the forced market exit


def load():
    B = pd.read_csv(DATA, parse_dates=['start', 'end'])
    C = B[(B.covered == True) & B.min_float.notna()].copy()   # noqa: E712
    C = C.sort_values('end').reset_index(drop=True)
    return C


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def metrics(net: np.ndarray, C: pd.DataFrame, label: str) -> dict:
    wins = net[net > 0]
    losses = net[net < 0]
    gp = wins.sum()
    gl = -losses.sum()
    equity = BALANCE + np.cumsum(net)          # closed-balance curve, ordered by close time
    dd_usc = max_drawdown(np.concatenate([[BALANCE], equity]))
    return dict(
        config=label,
        n_baskets=len(net),
        net_usc=round(float(net.sum()), 2),
        net_pct=round(100 * float(net.sum()) / BALANCE, 3),
        win_rate=round(100 * len(wins) / len(net), 2),
        profit_factor=round(float(gp / gl), 3) if gl > 0 else np.inf,
        worst_loss_usc=round(float(net.min()), 2),
        worst_loss_pct=round(100 * float(net.min()) / BALANCE, 3),
        max_dd_usc=round(dd_usc, 2),
        max_dd_pct=round(100 * dd_usc / BALANCE, 3),
    )


def main():
    C = load()
    print(f'tick-covered baskets loaded: {len(C)}')
    print(f'balance anchor: {BALANCE:,.2f} USC')
    print(f'period: {C.start.min()} .. {C.end.max()}\n')

    rows = []

    # ── baseline: as traded, no stop ───────────────────────────────────────
    base = C.net.values.astype(float)
    m = metrics(base, C, 'AS TRADED (no stop)')
    m['n_stopped'] = 0
    rows.append(m)

    # ── counterfactuals ───────────────────────────────────────────────────
    for s in STOPS:
        stop_usc = BALANCE * s
        hit = C.min_float.values <= -stop_usc
        # slippage on forced exit, proportional to the lots live at the stop.
        # Upper bound on live lots = total_lots (all rungs); using total_lots is
        # the pessimistic choice and is only applied to stopped baskets.
        slip = C.total_lots.values * 100.0 * SLIP_USD_PER_OZ    # USC (1 oz/lot, P/L in cents)
        net = np.where(hit, -stop_usc - slip, C.net.values.astype(float))
        m = metrics(net, C, f'STOP {s*100:.1f}% ({stop_usc:,.0f} USC)')
        m['n_stopped'] = int(hit.sum())
        rows.append(m)

    R = pd.DataFrame(rows)[['config', 'n_baskets', 'n_stopped', 'net_usc', 'net_pct',
                            'win_rate', 'profit_factor', 'worst_loss_usc',
                            'worst_loss_pct', 'max_dd_usc', 'max_dd_pct']]
    print(R.to_string(index=False))
    R.to_csv('data/gate0_stop_counterfactual.csv', index=False)

    # ── supporting diagnostics ────────────────────────────────────────────
    print('\n--- floating-loss distribution (min_float, USC) ---')
    print(C.min_float.describe([.1, .25, .5, .75, .9, .99]).to_string())
    print(f'\nbaskets never underwater at all: {(C.min_float >= 0).sum()}')
    print(f'baskets whose eventual profit was < their peak floating loss: '
          f'{((C.net > 0) & (C.net < -C.min_float)).sum()} / {(C.net > 0).sum()} winners')
    for s in STOPS:
        stop_usc = BALANCE * s
        hit = C.min_float.values <= -stop_usc
        rescued = C.net.values[hit]
        print(f'stop {s*100:>4.1f}%: {hit.sum():>3} stopped, '
              f'of which {(rescued > 0).sum()} were winners worth '
              f'{rescued[rescued > 0].sum():,.2f} USC that are now forgone')

    # ── zero-slippage sensitivity (idealised fills) ───────────────────────
    print('\n--- sensitivity: idealised fill exactly at the stop, zero slippage ---')
    for s in STOPS:
        stop_usc = BALANCE * s
        hit = C.min_float.values <= -stop_usc
        net = np.where(hit, -stop_usc, C.net.values.astype(float))
        print(f'stop {s*100:>4.1f}%: net {net.sum():>12,.2f} USC '
              f'({100*net.sum()/BALANCE:>7.3f}%)')


if __name__ == '__main__':
    main()
