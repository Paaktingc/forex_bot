"""
lever2_experiment.py  — Lever 2: probability calibration + ECDF/Kelly bet sizing.

Loads the cached prepared meta frame (data/_lever2_cache.pkl, produced by
run_pipeline.py with META_DUMP_CACHE=1) and re-runs the OFFICIAL
walk_forward_meta_labeling under combinations of:
  - probability_calibration : none | isotonic | sigmoid  (fit IN-SAMPLE per fold)
  - sizing_mode             : confidence | ecdf | kelly

For each combo it pulls the threshold-0.65 trade sequence (risk 0.5%), reports
WR/PF/Sharpe/return/realized-DD, and bootstrap Monte-Carlo P(maxDD > 4.5%) at
BOTH 0.5% sizing and a target-hitting sizing (sequence rescaled so total
return ~= 8%). RNG seed 42. Reproducible. Research only.
"""

from __future__ import annotations

import pickle

import numpy as np

from validation import walk_forward_meta_labeling

RANDOM_SEED = 42
HARD_DD_LIMIT = 0.045
TARGET_RETURN = 0.08
N_SIMS = 20_000
THRESHOLD = 0.65
RISK = 0.005


def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(-((equity - peak) / peak).min())


def _bootstrap_breach(pnl: np.ndarray, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    m = pnl.size
    dds = np.empty(n)
    for i in range(n):
        s = rng.choice(pnl, size=m, replace=True)
        eq = np.concatenate(([1.0], np.cumprod(1.0 + s)))
        dds[i] = _max_dd(eq)
    return float(np.mean(dds > HARD_DD_LIMIT)) * 100.0


def main() -> None:
    with open("data/_lever2_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)

    combos = [
        ("none", "confidence"),     # baseline (must reproduce official 77 / 4.94% / 3.36%)
        ("isotonic", "confidence"),
        ("sigmoid", "confidence"),
        ("none", "ecdf"),
        ("none", "kelly"),
        ("isotonic", "kelly"),
        ("sigmoid", "kelly"),
    ]

    print(f"{'calib':9} {'sizing':11} {'n':>4} {'WR%':>5} {'PF':>5} {'Sharpe':>7} "
          f"{'Ret%':>7} {'DD%':>6} {'P>4.5@0.5%':>11} {'P>4.5@tgt':>10}")
    print("-" * 92)

    for calib, sizing in combos:
        res = walk_forward_meta_labeling(
            cache["df_model"], cache["selected_feature_cols"],
            train_months=6, test_months=1,
            purge_gap=cache["purge_gap"], embargo_gap=cache["embargo_gap"],
            risk_per_trade=RISK, thresholds=cache["thresholds"],
            meta_config=cache["meta_config"],
            probability_calibration=calib, sizing_mode=sizing,
        )
        trades = res["filtered_trades"][THRESHOLD]
        if not trades:
            print(f"{calib:9} {sizing:11}  no trades")
            continue
        m = res["best_threshold_result"] if res.get("best_threshold") == THRESHOLD else None
        # always recompute from the 0.65 trades for consistency
        pnl = np.array([t["pct_return"] for t in trades], dtype=float)
        eq = np.concatenate(([1.0], np.cumprod(1.0 + pnl)))
        ret = (eq[-1] - 1.0) * 100
        dd = _max_dd(eq) * 100
        thr_res = next(r for r in res["threshold_results"] if r["threshold"] == THRESHOLD)
        wr = float(thr_res["win_rate_value"]) * 100
        pf = float(thr_res["profit_factor_value"])
        sharpe = float(thr_res["sharpe_ratio_value"])

        p_half = _bootstrap_breach(pnl, N_SIMS, RANDOM_SEED)
        scale = (TARGET_RETURN / (ret / 100)) if ret > 0 else float("nan")
        p_tgt = _bootstrap_breach(pnl * scale, N_SIMS, RANDOM_SEED) if np.isfinite(scale) else float("nan")

        print(f"{calib:9} {sizing:11} {pnl.size:>4} {wr:>5.1f} {pf:>5.2f} {sharpe:>7.2f} "
              f"{ret:>7.2f} {dd:>6.2f} {p_half:>10.2f}% {p_tgt:>9.2f}%")


if __name__ == "__main__":
    main()
