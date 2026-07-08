"""
monte_carlo_dd.py

Monte Carlo analysis for The5ers Bootcamp step evaluation.

Core pieces (imported by backtest.py and the tests):
  - simulate_step():       walks one per-trade return path through a Bootcamp
                           step simulator that absorbs at +6% (PASS) / −5%
                           (BREACH), with the bot's −3% kill switch active
                           (KILL ends the path before the official breach).
  - block_bootstrap_paths(): ≥20,000 block-bootstrap resamples (block ≈ 10
                           trades — preserves local streak structure).
  - run_step_monte_carlo(): paths → outcome probabilities + trade stats
                           (PF, max DD, avg R, win rate, expectancy, trades,
                           longest losing streak, P(breach), P(kill),
                           P(pass before fail), median trades-to-pass).

Legacy CLI (per-trade CSV with a `pct_return` column) still works:
    python monte_carlo_dd.py [trades_csv] [n_sims]

Reproducible: RNG seeded with RANDOM_SEED.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# The5ers Bootcamp step geometry (fractions of the initial step balance)
STEP_TARGET = 0.06        # +6% → step passed
STEP_FAIL = -0.05         # −5% static → step failed (official limit)
KILL_SWITCH = -0.03       # bot's operative halt (must fire before −5%)

DEFAULT_N_PATHS = 20_000
DEFAULT_BLOCK_SIZE = 10


# ---------------------------------------------------------------------------
# Step simulator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepResult:
    outcome: str              # "PASS" | "KILL" | "BREACH" | "INCOMPLETE"
    trades_used: int
    final_equity: float       # relative to 1.0 step-start balance


def simulate_step(
    returns: np.ndarray,
    target: float = STEP_TARGET,
    fail: float = STEP_FAIL,
    kill: float = KILL_SWITCH,
) -> StepResult:
    """
    Walks compounded equity (start 1.0) through per-trade fractional returns.
    Absorbs at +target (PASS) or −fail (BREACH); the kill switch ends the
    path at −kill (KILL) — trading stops there, so the official −5% is only
    reachable when a single trade gaps straight through it.
    """
    equity = 1.0
    for i, r in enumerate(np.asarray(returns, dtype=float), start=1):
        equity *= 1.0 + r
        if equity <= 1.0 + fail:
            return StepResult("BREACH", i, equity)
        if equity <= 1.0 + kill:
            return StepResult("KILL", i, equity)
        if equity >= 1.0 + target:
            return StepResult("PASS", i, equity)
    return StepResult("INCOMPLETE", len(returns), equity)


def block_bootstrap_paths(
    returns: np.ndarray,
    n_paths: int = DEFAULT_N_PATHS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    path_len: int | None = None,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Block bootstrap: concatenates blocks of ``block_size`` CONSECUTIVE trades
    (sampled with replacement) so streak/clustering structure survives the
    resampling. Returns an (n_paths, path_len) matrix of per-trade returns.
    """
    returns = np.asarray(returns, dtype=float)
    m = returns.size
    if m == 0:
        raise ValueError("block_bootstrap_paths: no trades to resample")
    block_size = max(1, min(block_size, m))
    if path_len is None:
        path_len = max(300, 3 * m)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(path_len / block_size))
    # start index of each block (wrap-around circular bootstrap)
    starts = rng.integers(0, m, size=(n_paths, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % m
    paths = returns[idx.reshape(n_paths, -1)[:, :path_len]]
    return paths


def _step_outcomes_vectorized(
    paths: np.ndarray,
    target: float = STEP_TARGET,
    fail: float = STEP_FAIL,
    kill: float = KILL_SWITCH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized step simulation over a (n_paths, path_len) return matrix.
    Returns (outcome_codes, trades_to_outcome); codes: 0 PASS, 1 KILL,
    2 BREACH, 3 INCOMPLETE.
    """
    equity = np.cumprod(1.0 + paths, axis=1)
    n_paths, path_len = equity.shape
    big = path_len + 1

    def first_index(mask: np.ndarray) -> np.ndarray:
        any_hit = mask.any(axis=1)
        first = np.where(any_hit, mask.argmax(axis=1), big)
        return first

    idx_pass = first_index(equity >= 1.0 + target)
    idx_kill = first_index(equity <= 1.0 + kill)   # includes breach levels
    idx_breach = first_index(equity <= 1.0 + fail)

    outcomes = np.full(n_paths, 3, dtype=int)          # INCOMPLETE
    trades = np.full(n_paths, path_len, dtype=int)

    kill_first = idx_kill < idx_pass
    pass_first = idx_pass <= idx_kill

    # Path absorbed downward: BREACH if that same trade already gapped
    # through −5%, otherwise the kill switch stopped it at −3%.
    breach_mask = kill_first & (idx_breach == idx_kill) & (idx_kill <= path_len)
    kill_mask = kill_first & ~breach_mask & (idx_kill <= path_len)
    pass_mask = pass_first & (idx_pass <= path_len)

    outcomes[pass_mask] = 0
    outcomes[kill_mask] = 1
    outcomes[breach_mask] = 2
    trades[pass_mask] = idx_pass[pass_mask] + 1
    trades[kill_mask] = idx_kill[kill_mask] + 1
    trades[breach_mask] = idx_breach[breach_mask] + 1
    return outcomes, trades


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------

def longest_losing_streak(returns: np.ndarray) -> int:
    streak = best = 0
    for r in returns:
        if r < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def trade_stats(pct_returns: np.ndarray, r_multiples: np.ndarray | None = None) -> dict:
    """PF, max DD, avg R, win rate, expectancy, trades, longest losing streak."""
    pnl = np.asarray(pct_returns, dtype=float)
    if pnl.size == 0:
        return {
            "trades": 0, "profit_factor": 0.0, "win_rate_pct": 0.0,
            "avg_r": 0.0, "expectancy_pct": 0.0, "max_drawdown_pct": 0.0,
            "longest_losing_streak": 0,
        }

    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = float("inf") if losses == 0 and gains > 0 else (gains / losses if losses else 0.0)

    equity = np.concatenate(([1.0], np.cumprod(1.0 + pnl)))
    peaks = np.maximum.accumulate(equity)
    max_dd = float(((peaks - equity) / peaks).max())

    r = np.asarray(r_multiples, dtype=float) if r_multiples is not None else None
    return {
        "trades": int(pnl.size),
        "profit_factor": round(float(pf), 4) if np.isfinite(pf) else float("inf"),
        "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 2),
        "avg_r": round(float(np.nanmean(r)), 4) if r is not None and r.size else 0.0,
        "expectancy_pct": round(float(pnl.mean() * 100.0), 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "longest_losing_streak": longest_losing_streak(pnl),
    }


# ---------------------------------------------------------------------------
# Full Monte Carlo through the step simulator
# ---------------------------------------------------------------------------

def run_step_monte_carlo(
    pct_returns: np.ndarray,
    r_multiples: np.ndarray | None = None,
    n_paths: int = DEFAULT_N_PATHS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Block-bootstraps ``n_paths`` (≥20k) trade sequences and pushes each
    through the Bootcamp step simulator. Returns trade stats + outcome
    probabilities used by the GO/NO-GO verdict.
    """
    paths = block_bootstrap_paths(
        pct_returns, n_paths=n_paths, block_size=block_size, seed=seed
    )
    outcomes, trades = _step_outcomes_vectorized(paths)

    n = float(len(outcomes))
    p_pass = float((outcomes == 0).sum()) / n
    p_kill = float((outcomes == 1).sum()) / n
    p_breach = float((outcomes == 2).sum()) / n
    p_incomplete = float((outcomes == 3).sum()) / n

    pass_trades = trades[outcomes == 0]
    stats = trade_stats(pct_returns, r_multiples)
    stats.update(
        {
            "n_paths": int(n_paths),
            "block_size": int(block_size),
            "p_pass": round(p_pass, 4),
            "p_kill_switch": round(p_kill + p_breach, 4),  # kill fires by −3% incl. gaps
            "p_breach_official": round(p_breach, 4),
            "p_incomplete": round(p_incomplete, 4),
            "median_trades_to_pass": float(np.median(pass_trades)) if pass_trades.size else float("nan"),
        }
    )
    return stats


def print_step_report(result: dict) -> None:
    print(f"\nMONTE CARLO — Bootcamp step simulator "
          f"({result['n_paths']:,} block-bootstrap paths, block={result['block_size']})")
    print("-" * 60)
    print(f"Trades (source):          {result['trades']}")
    print(f"Profit factor:            {result['profit_factor']}")
    print(f"Win rate:                 {result['win_rate_pct']:.2f}%")
    print(f"Avg R:                    {result['avg_r']}")
    print(f"Expectancy / trade:       {result['expectancy_pct']:.4f}%")
    print(f"Max drawdown (realized):  {result['max_drawdown_pct']:.2f}%")
    print(f"Longest losing streak:    {result['longest_losing_streak']}")
    print(f"P(pass +6% before −5%):   {result['p_pass']*100:.2f}%")
    print(f"P(hit −3% kill switch):   {result['p_kill_switch']*100:.2f}%")
    print(f"P(breach −5% official):   {result['p_breach_official']*100:.2f}%")
    print(f"P(incomplete path):       {result['p_incomplete']*100:.2f}%")
    if result["median_trades_to_pass"] == result["median_trades_to_pass"]:
        print(f"Median trades to pass:    {result['median_trades_to_pass']:.0f}")


# ---------------------------------------------------------------------------
# Legacy CLI (kept for existing per-trade CSVs)
# ---------------------------------------------------------------------------

def _max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough max drawdown (fraction, >=0) of a compounded equity path."""
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (equity - running_peak) / running_peak
    return float(-drawdowns.min())


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/meta_trades_thr0.65.csv"
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_PATHS

    df = pd.read_csv(csv_path)
    if "pct_return" not in df.columns:
        raise SystemExit(f"{csv_path} has no 'pct_return' column. Columns: {list(df.columns)}")
    pnl = df["pct_return"].astype(float).to_numpy()

    realized_equity = np.concatenate(([1.0], np.cumprod(1.0 + pnl)))
    print(f"Loaded {pnl.size} trades from {csv_path}")
    print(f"Realized (single-path) max DD : {_max_drawdown(realized_equity)*100:.2f}%")
    print(f"Realized (single-path) return : {(realized_equity[-1]-1.0)*100:.2f}%")
    print(f"RNG seed: {RANDOM_SEED} | paths: {n_sims:,}")

    result = run_step_monte_carlo(pnl, n_paths=n_sims)
    print_step_report(result)


if __name__ == "__main__":
    main()
