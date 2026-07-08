"""
monte_carlo_dd.py

Monte Carlo tail-drawdown analysis on a persisted per-trade P&L sequence.

Reads a CSV of trades (expects a `pct_return` column = per-trade account-%
return, the same field walk_forward_meta_labeling uses to build equity), then
resamples the sequence many times to estimate the distribution of max drawdown
and total return. Quantifies the probability that real trade sequencing breaches
The5ers' 4.5% absolute drawdown limit.

Two resampling schemes:
  - bootstrap  : sample WITH replacement (varies both the trade multiset & order)
  - permutation: shuffle WITHOUT replacement (fixed trade set, varies only order)

Reproducible: RNG seeded with RANDOM_SEED.

Usage:
    python monte_carlo_dd.py [trades_csv] [n_sims]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

RANDOM_SEED = 42
HARD_DD_LIMIT = 0.045      # The5ers absolute max drawdown
DAILY_DD_LIMIT = 0.040     # The5ers daily loss limit (used as a stricter DD proxy)


def _max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough max drawdown (fraction, >=0) of a compounded equity path."""
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (equity - running_peak) / running_peak
    return float(-drawdowns.min())


def _simulate(pnl: np.ndarray, n_sims: int, *, replace: bool, seed: int):
    """
    Returns (max_dd_array, total_return_array) over n_sims resamples.
    Equity compounds as prod(1 + r), matching validation._summarise_trades.
    """
    rng = np.random.default_rng(seed)
    m = pnl.size
    max_dds = np.empty(n_sims, dtype=float)
    total_returns = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        if replace:
            sample = rng.choice(pnl, size=m, replace=True)
        else:
            sample = pnl.copy()
            rng.shuffle(sample)
        equity = np.cumprod(1.0 + sample)
        equity = np.concatenate(([1.0], equity))  # anchor at 1.0
        max_dds[i] = _max_drawdown(equity)
        total_returns[i] = equity[-1] - 1.0

    return max_dds, total_returns


def _report(name: str, max_dds: np.ndarray, total_returns: np.ndarray) -> None:
    pct = lambda a, q: float(np.percentile(a, q)) * 100.0
    print(f"\n=== {name} (n={max_dds.size:,}) ===")
    print("Max-drawdown distribution:")
    print(f"  mean    : {max_dds.mean()*100:6.2f}%")
    print(f"  median  : {np.median(max_dds)*100:6.2f}%")
    print(f"  90th pct: {pct(max_dds,90):6.2f}%")
    print(f"  95th pct: {pct(max_dds,95):6.2f}%")
    print(f"  99th pct: {pct(max_dds,99):6.2f}%")
    print(f"  worst   : {max_dds.max()*100:6.2f}%")
    p_hard = float(np.mean(max_dds > HARD_DD_LIMIT)) * 100.0
    p_daily = float(np.mean(max_dds > DAILY_DD_LIMIT)) * 100.0
    print(f"  P(maxDD > 4.5% absolute limit): {p_hard:5.2f}%")
    print(f"  P(maxDD > 4.0% daily-equiv proxy): {p_daily:5.2f}%")
    print("Total-return distribution:")
    print(f"  mean    : {total_returns.mean()*100:6.2f}%")
    print(f"  median  : {np.median(total_returns)*100:6.2f}%")
    print(f"  5th pct : {pct(total_returns,5):6.2f}%")
    print(f"  95th pct: {pct(total_returns,95):6.2f}%")
    p_neg = float(np.mean(total_returns < 0)) * 100.0
    print(f"  P(return < 0): {p_neg:5.2f}%")


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/meta_trades_thr0.65.csv"
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000

    df = pd.read_csv(csv_path)
    if "pct_return" not in df.columns:
        raise SystemExit(f"{csv_path} has no 'pct_return' column. Columns: {list(df.columns)}")
    pnl = df["pct_return"].astype(float).to_numpy()

    realized_equity = np.concatenate(([1.0], np.cumprod(1.0 + pnl)))
    print(f"Loaded {pnl.size} trades from {csv_path}")
    print(f"Realized (single-path) max DD : {_max_drawdown(realized_equity)*100:.2f}%")
    print(f"Realized (single-path) return : {(realized_equity[-1]-1.0)*100:.2f}%")
    print(f"Per-trade pct_return: mean={pnl.mean()*100:.4f}% std={pnl.std()*100:.4f}% "
          f"min={pnl.min()*100:.4f}% max={pnl.max()*100:.4f}%")
    print(f"RNG seed: {RANDOM_SEED} | simulations per scheme: {n_sims:,}")

    boot_dd, boot_ret = _simulate(pnl, n_sims, replace=True, seed=RANDOM_SEED)
    perm_dd, perm_ret = _simulate(pnl, n_sims, replace=False, seed=RANDOM_SEED)
    _report("BOOTSTRAP (with replacement)", boot_dd, boot_ret)
    _report("PERMUTATION (without replacement)", perm_dd, perm_ret)


if __name__ == "__main__":
    main()
