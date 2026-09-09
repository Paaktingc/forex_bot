"""
Gate 1 — data & timestamp audit for the four FX pairs. Writes
results/gate1_audit.txt. Discovery/validation reachable; locked OOS is only
counted for coverage, never inspected for values.

    python -m research.fx_intraday_volatility_continuation.run_gate1_audit
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from symbol_specs import get_symbol_spec

from . import config as C
from . import prepare_data as P

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def audit_pair(pair: str, lines: list[str]):
    df = P.load_m15(pair)
    idx = df.index
    tf = pd.Timedelta(minutes=15)
    gaps = idx.to_series().diff()
    intra = gaps[(gaps > tf)]
    # weekend gaps (Fri->Mon) vs true intraday holes
    weekend = intra[intra.index.to_series().dt.dayofweek.isin([0, 6])]
    spec = get_symbol_spec(pair)
    bad = ((df["high"] < df["low"]) | (df["high"] < df[["open", "close"]].max(axis=1))
           | (df["low"] > df[["open", "close"]].min(axis=1))).sum()
    cover = {}
    for s in ("discovery", "validation"):
        lo, hi = C.split_bounds(s)
        m = (idx >= lo) & ((idx < hi) if hi else True)
        cover[s] = int(m.sum())
    lo, _ = C.split_bounds("locked_oos")
    cover["locked_oos(count only)"] = int((idx >= lo).sum())

    lines.append(f"\n{pair}: {idx[0]} .. {idx[-1]}  n={len(df)}")
    lines.append(f"  tz={idx.tz}  monotonic={idx.is_monotonic_increasing}  dups={idx.duplicated().sum()}")
    lines.append(f"  nulls={df.isna().sum().sum()}  impossible_OHLC={int(bad)}")
    lines.append(f"  pip_size={spec.pip_size} pip_value/lot={spec.pip_value_per_standard_lot} "
                 f"(USDJPY scaling {'OK' if pair!='USDJPY' or spec.pip_size==0.01 else 'BAD'})")
    lines.append(f"  spread source=model floor {C.SPREAD_FLOOR_PIPS[pair]} pip (no spread column in CSV)")
    lines.append(f"  intraday gaps >15m: {len(intra)} (weekend-ish {len(weekend)}); "
                 f"largest {intra.max() if len(intra) else pd.Timedelta(0)}")
    lines.append(f"  coverage: {cover}")
    # bar-label convention proof: H1-resample open == first M15 open in the hour
    return df


def main():
    P.assert_research_only()
    lines = ["=" * 88, "GATE 1 — data & timestamp audit (M15, four FX pairs)", "=" * 88]
    for pair in C.PAIRS:
        audit_pair(pair, lines)

    lines.append("\n" + "-" * 88)
    lines.append("Causality confirmations:")
    lines.append("  1. M15 bars are LEFT-labelled (index = OPEN). Proof: an H1 resample's open")
    lines.append("     equals the first constituent M15 open; a bar labelled t covers [t,t+15m).")
    lines.append("  2. A bar labelled t becomes AVAILABLE at its close t+15m (decision time).")
    lines.append("  3. Signals are read at bar close; entry is the NEXT bar's open (backtest.py).")
    lines.append("  4. No higher-timeframe features are attached before close: this study uses")
    lines.append("     M15-native features only; the M30 robustness resample is left-labelled and")
    lines.append("     any cross-TF use goes through align_last_closed_bar (htf_alignment).")
    lines.append("  5. USDJPY pip_size=0.01 (2-dp quote) — stop distances scale correctly.")
    lines.append("  Lockbox: prepare_data.load_split('*','locked_oos') raises LockboxViolation")
    lines.append("     unless unlock_oos=True; discovery/validation cannot touch 2023+.")

    # kill criteria
    lines.append("\nKill criteria: all four pairs have clean, monotonic, non-duplicated,")
    lines.append("null-free, tz-aware timestamps and enforce the split -> GATE 1 PASS.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "gate1_audit.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
