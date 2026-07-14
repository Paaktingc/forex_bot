"""
research_ablation.py

Phase-2 component ablation for the pullback framework (design window only).
Same exits (SL 1.5×ATR clamped 8–25 pips, TP 2R, BE 1R) and same costs for
all variants; sequential one-at-a-time entries, London session only, no
pacing (comparability):

  a) regime-only baseline — first valid in-session bar each day within
     regime, market entry, SL anchored at entry (no swing)
  b) regime + pullback zone with a LIMIT order at the M15 EMA20 (no
     confirmation close, no RSI); limit fills pay spread but no slippage
  c) current full trigger (pullback + confirmation close + RSI recross),
     market entry

Usage:
    python research_ablation.py [--lockbox-start 2025-03-20]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import backtest
import config
import strategy as strategy_module
from risk_manager import compute_sl_tp


def _prepared_engine(d15: pd.DataFrame, dh1: pd.DataFrame) -> backtest.RulesBacktestEngine:
    eng = backtest.RulesBacktestEngine(d15, dh1)
    df = eng.df
    eng._opens = df["open"].to_numpy(float)
    eng._highs = df["high"].to_numpy(float)
    eng._lows = df["low"].to_numpy(float)
    eng._closes = df["close"].to_numpy(float)
    eng._news_bar_mask = eng._news_mask(df.index, config.NEWS_BUFFER_MINS)
    return eng


def _simulate(eng, entries) -> pd.DataFrame:
    """
    entries: iterable of (i_entry, direction, entry_price, swing_or_None).
    Sequential one-at-a-time; SL/TP via the production clamp; returns per-trade
    R after costs (walker applies commission; entry costs are already in
    entry_price).
    """
    rows = []
    busy_until = -1
    atrs = eng._atrs
    for i, sig, entry, swing in entries:
        if i <= busy_until:
            continue
        atr = float(atrs[i - 1])
        if not np.isfinite(atr) or atr <= 0:
            continue
        sl_tp = compute_sl_tp(sig, entry, atr, swing_price=swing)
        if sl_tp is None:
            continue
        sl, tp = sl_tp
        risk = (entry - sl) * sig
        out = eng._walk_exit(i, sig, entry, sl, tp, risk, use_be=True)
        busy_until = out["exit_idx"]
        rows.append(
            {"entry_time": eng.df.index[i], "direction": sig, "r": out["r"],
             "exit_reason": out["exit_reason"]}
        )
    return pd.DataFrame(rows)


def run_ablation(d15: pd.DataFrame, dh1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eng = _prepared_engine(d15, dh1)
    frame = strategy_module.build_signal_frame(eng.df, eng.df_h1, eng.params)
    eng._atrs = frame["atr_14"].to_numpy(float)

    index = eng.df.index
    session_ok = np.array([strategy_module.entry_session_ok(ts) for ts in index])
    regime = frame["regime"].to_numpy(int)
    ema_pull = frame["ema_pull"].to_numpy(float)
    swing_low = frame["swing_low"].to_numpy(float)
    swing_high = frame["swing_high"].to_numpy(float)
    signals = frame["signal"].to_numpy(int)
    opens = eng._opens
    lows, highs = eng._lows, eng._highs
    entry_cost = eng.spread + eng.slip_entry

    # (a) regime-only: first in-session bar of each day with a regime
    entries_a = []
    last_day = None
    for i in range(1, len(index)):
        day = index[i].date()
        if day == last_day or not session_ok[i] or regime[i - 1] == 0:
            continue
        sig = int(regime[i - 1])
        entries_a.append((i, sig, opens[i] + sig * entry_cost, None))
        last_day = day

    # (b) regime + limit at the EMA20 zone (no confirmation, no RSI):
    #     fill when the bar trades through the previous bar's EMA20
    entries_b = []
    for i in range(1, len(index)):
        if not session_ok[i] or regime[i - 1] == 0:
            continue
        sig = int(regime[i - 1])
        zone = ema_pull[i - 1]
        if not np.isfinite(zone):
            continue
        touched = lows[i] <= zone if sig == 1 else highs[i] >= zone
        if not touched:
            continue
        entry = zone + sig * eng.spread  # limit fill: spread, no slippage
        swing = swing_low[i - 1] if sig == 1 else swing_high[i - 1]
        entries_b.append((i, sig, entry, float(swing) if np.isfinite(swing) else None))

    # (c) full trigger, market entry
    entries_c = []
    for i in range(1, len(index)):
        sig = int(signals[i - 1])
        if sig == 0 or not session_ok[i]:
            continue
        swing = swing_low[i - 1] if sig == 1 else swing_high[i - 1]
        entries_c.append((i, sig, opens[i] + sig * entry_cost,
                          float(swing) if np.isfinite(swing) else None))

    return {
        "a_regime_only": _simulate(eng, entries_a),
        "b_zone_limit": _simulate(eng, entries_b),
        "c_full_trigger": _simulate(eng, entries_c),
    }


def report(results: dict[str, pd.DataFrame], fold_months: int = 6) -> None:
    def stats(df: pd.DataFrame) -> str:
        if df.empty:
            return "no trades"
        r = df["r"]
        gains, losses = r[r > 0].sum(), -r[r < 0].sum()
        pf = gains / losses if losses else float("inf")
        return (f"n={len(r):>5}  PF={pf:5.3f}  exp={r.mean():+.4f}R  "
                f"WR={(r > 0).mean()*100:5.1f}%")

    print(f"{'variant':16s}  overall")
    for name, df in results.items():
        print(f"{name:16s}  {stats(df)}")

    # per half-year fold
    print("\nPer 6-month fold (PF / expectancy R / n):")
    def _half(ts: pd.Timestamp) -> str:
        return f"{ts.year}H{1 if ts.month <= 6 else 2}"

    periods = sorted(
        {_half(t) for df in results.values() if not df.empty for t in df.entry_time}
    )
    print("period   " + "".join(f"{n[:16]:>26s}" for n in results))
    for period in periods:
        row = f"{period:8s} "
        for df in results.values():
            if df.empty:
                row += f"{'—':>26s}"
                continue
            r = df.loc[[_half(t) == period for t in df.entry_time], "r"]
            if len(r) == 0:
                row += f"{'—':>26s}"
                continue
            gains, losses = r[r > 0].sum(), -r[r < 0].sum()
            pf = gains / losses if losses > 0 else float("inf")
            row += f"   {pf:6.2f} {r.mean():+.3f}R n={len(r):<4}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lockbox-start", default="2025-03-20")
    args = parser.parse_args()

    lockbox = pd.Timestamp(args.lockbox_start, tz="UTC")
    df15 = backtest.get_ohlcv_from_csv("EURUSD", "M15")
    dfh1 = backtest.get_ohlcv_from_csv("EURUSD", "H1")
    results = run_ablation(df15[df15.index < lockbox], dfh1[dfh1.index < lockbox])
    report(results)
    for name, df in results.items():
        df.to_csv(f"/tmp/ablation_{name}.csv", index=False)


if __name__ == "__main__":
    main()
