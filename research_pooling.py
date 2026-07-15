"""
research_pooling.py

Cycle-2 Lever B: pool per-symbol regime_daily trade streams under live
constraints — ONE open trade globally, global daily/weekly pacing stops,
0.3% risk per trade on the pooled balance. Per-trade net return is taken
from each engine's pct_return (≈ 0.003 × net R), re-based multiplicatively
on the pooled equity path.

Approximations (documented): trades are admitted in entry-time order and
their results settle at exit time for busy-blocking; the consecutive-loss
counters update in entry order. Symbol ties break alphabetically.

Usage:
    python research_pooling.py SYMBOL [SYMBOL ...]   (reads per-pair CSVs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest
import config

RISK_PCT = 0.003


def collect_trades(symbols, start=None, end=None) -> list[dict]:
    """Runs the production engine per symbol and returns all trades."""
    all_trades: list[dict] = []
    for sym in symbols:
        df15 = backtest.get_ohlcv_from_csv(sym, "M15")
        dfh1 = backtest.get_ohlcv_from_csv(sym, "H1")
        if start is not None:
            # keep warm-up context before `start`; trades filtered later
            df15 = df15[df15.index >= start - pd.DateOffset(days=30)]
            dfh1 = dfh1[dfh1.index < (end or df15.index[-1] + pd.Timedelta("1d"))]
        if end is not None:
            df15 = df15[df15.index < end]
            dfh1 = dfh1[dfh1.index < end]
        eng = backtest.RulesBacktestEngine(df15, dfh1, symbol=sym)
        eng.run()
        for t in eng.trades:
            if start is not None and t["entry_time"] < start:
                continue
            all_trades.append({**t, "symbol": sym})
    return all_trades


def pool_trades(trades: list[dict]) -> pd.DataFrame:
    """
    Merges per-symbol streams under the live global constraints and returns
    the accepted trades with pooled pct returns, in entry order.
    """
    ordered = sorted(trades, key=lambda t: (t["entry_time"], t["symbol"]))

    busy_until = None
    balance = 1.0
    day = week = None
    day_start = week_start = balance
    trades_today = 0
    consec_day = 0
    consec_week = 0
    halted_today = halted_week = False

    rows = []
    for t in ordered:
        ts = t["entry_time"]
        if day != ts.date():
            day = ts.date()
            day_start = balance
            trades_today = 0
            consec_day = 0
            halted_today = False
        if week != ts.isocalendar()[:2]:
            week = ts.isocalendar()[:2]
            week_start = balance
            consec_week = 0
            halted_week = False

        if busy_until is not None and ts <= busy_until:
            continue
        if halted_today or halted_week:
            continue
        if trades_today >= config.MAX_TRADES_PER_DAY:
            continue
        if consec_day >= config.MAX_CONSEC_LOSSES_DAY:
            halted_today = True
            continue
        if consec_week >= config.MAX_CONSEC_LOSSES_WEEK:
            halted_week = True
            continue
        if (day_start - balance) / day_start >= config.DAILY_LOSS_PCT:
            halted_today = True
            continue
        if (week_start - balance) / week_start >= config.WEEKLY_STOP_PCT:
            halted_week = True
            continue

        net_r = t["pct_return"] / RISK_PCT     # engine risked 0.3%/trade
        pct = RISK_PCT * net_r
        balance *= 1.0 + pct
        busy_until = t["exit_time"]
        trades_today += 1
        if pct < 0:
            consec_day += 1
            consec_week += 1
        elif pct > 0:
            consec_day = consec_week = 0

        rows.append({
            "entry_time": ts, "exit_time": t["exit_time"], "symbol": t["symbol"],
            "pct_return": pct, "net_r": net_r, "exit_reason": t["exit_reason"],
        })

    return pd.DataFrame(rows)


def fold_table(pooled: pd.DataFrame, fold_starts: list[pd.Timestamp],
               months: int = 6) -> list[dict]:
    folds = []
    for i, fs in enumerate(fold_starts, 1):
        fe = fs + pd.DateOffset(months=months)
        r = pooled.loc[(pooled.entry_time >= fs) & (pooled.entry_time < fe), "pct_return"]
        if len(r) == 0:
            continue
        g, l = r[r > 0].sum(), -r[r < 0].sum()
        folds.append({
            "fold": i, "start": str(fs.date()), "n": len(r),
            "pf": g / l if l > 0 else float("inf"),
            "avg_r": pooled.loc[r.index, "net_r"].mean(),
        })
    return folds


def summarize(pooled: pd.DataFrame, label: str) -> None:
    r = pooled["pct_return"].to_numpy()
    eq = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peaks = np.maximum.accumulate(eq)
    maxdd = ((peaks - eq) / peaks).max() * 100
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    print(f"{label}: n={len(r)} PF={g/l:0.4f} WR={(r>0).mean()*100:.1f}% "
          f"ret={(eq[-1]-1)*100:+.1f}% maxDD={maxdd:.2f}%")


if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    lockbox = pd.Timestamp("2025-03-20", tz="UTC")
    trades = collect_trades(symbols, end=lockbox)
    pooled = pool_trades(trades)
    summarize(pooled, f"POOL[{'+'.join(symbols)}] design window")
