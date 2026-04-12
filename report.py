"""
report.py

Change summary:
- Added dedicated reporting for raw base signals versus meta-model filtered
  signals.
- Shows threshold-by-threshold performance and highlights the best probability
  cutoff by risk-adjusted return.
"""

from __future__ import annotations


def _metric_line(label: str, metrics: dict, key: str) -> str:
    return f"{label:<26} {metrics.get(key, 'n/a')}"


def _stat_line(label: str, stats: dict) -> str:
    return f"{label:<26} mean={stats.get('mean', 'n/a')} | median={stats.get('median', 'n/a')} | std={stats.get('std', 'n/a')}"


def generate_backtest_report(wf_results: dict, title: str = "META-LABELING BACKTEST") -> str:
    """
    Pretty-print base strategy versus meta-filtered strategy performance.
    """
    base = wf_results.get("base_strategy", {})
    best = wf_results.get("best_threshold_result", {})
    threshold_results = wf_results.get("threshold_results", [])

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Validation Verdict:        {wf_results.get('verdict', 'UNKNOWN')}")
    lines.append(f"Readiness:                 {wf_results.get('readiness', 'NOT READY')}")
    lines.append(f"Checks Passed:             {wf_results.get('checks_passed', '0/8')}")
    lines.append(f"Best Threshold:            {wf_results.get('best_threshold', 'n/a')}")
    lines.append("")
    lines.append("--- Target Definition ---")
    lines.append(f"Target type:               {wf_results.get('target_type', 'meta-labeling')}")
    lines.append(f"Definition:                {wf_results.get('target_definition', 'n/a')}")
    lines.append("")
    lines.append("--- Causality Boundary ---")
    lines.append("Features use:              signal bar t and earlier only")
    lines.append("Execution uses:            open[t+1]")
    lines.append(f"Purge gap:                 {wf_results.get('purge_gap', 'n/a')} bars")
    lines.append(f"Embargo gap:               {wf_results.get('embargo_gap', 'n/a')} bars")
    max_corr = wf_results.get("max_feature_target_corr", "n/a")
    if isinstance(max_corr, (float, int)):
        max_corr = f"{float(max_corr):.3f}"
    lines.append(f"Feature-target max corr:   {max_corr}")
    lines.append(f"Leakage risk:              {wf_results.get('leakage_risk', 'UNKNOWN')}")
    lines.append("")
    regime_diag = wf_results.get("regime_diagnostic")
    if regime_diag:
        lines.append("--- Regime Diagnostic ---")
        lines.append(f"Conditional edge found:    {regime_diag.get('conditional_edge_found', False)}")
        lines.append(f"Edge buckets flagged:      {regime_diag.get('edge_bucket_count', 0)}")
        for bucket in regime_diag.get("edge_buckets", [])[:5]:
            lines.append(
                f"  - [{bucket.get('dimension', 'n/a')}] {bucket.get('bucket', 'n/a')} | "
                f"WR={bucket.get('win_rate', 'n/a')}% | PF={bucket.get('profit_factor', 'n/a')} | "
                f"n={bucket.get('trades', 'n/a')}"
            )
        lines.append("")
    importance_diag = wf_results.get("feature_importance")
    if importance_diag:
        lines.append("--- Feature Selection ---")
        lines.append(f"Candidate features:        {importance_diag.get('candidate_feature_count', 'n/a')}")
        lines.append(f"Selected features:         {importance_diag.get('selected_feature_count', 'n/a')}")
        lines.append(f"Noise threshold:           {importance_diag.get('noise_threshold', 'n/a')}")
        lines.append(f"Noise median:              {importance_diag.get('noise_median', 'n/a')}")
        shuffle_old = importance_diag.get("shuffle_accuracy_legacy", "n/a")
        shuffle_balanced = importance_diag.get("shuffle_accuracy_balanced", "n/a")
        if isinstance(shuffle_old, (float, int)):
            shuffle_old = f"{float(shuffle_old) * 100:.1f}%"
        if isinstance(shuffle_balanced, (float, int)):
            shuffle_balanced = f"{float(shuffle_balanced) * 100:.1f}%"
        lines.append(f"Legacy shuffle accuracy:   {shuffle_old}")
        lines.append(f"Balanced shuffle accuracy: {shuffle_balanced}")
        lines.append(f"Leakage suspected:         {importance_diag.get('leakage_suspected', False)}")
        selected_features = importance_diag.get("selected_features", [])
        if selected_features:
            lines.append("Selected feature list:")
            for feature in selected_features:
                lines.append(f"  - {feature}")
        lines.append("")
    lines.append("--- Model Diagnostics ---")
    lines.append(f"Avg IS Accuracy:           {wf_results.get('avg_is_accuracy', 'n/a')}")
    lines.append(f"Avg OOS Accuracy:          {wf_results.get('avg_oos_accuracy', 'n/a')}")
    lines.append(f"Overfit Ratio (IS/OOS):    {wf_results.get('overfit_ratio', 'n/a')}")
    lines.append("")
    lines.append("--- Base Strategy (Raw Signals) ---")
    lines.append(_metric_line("OOS Return", base, "total_return"))
    lines.append(_metric_line("Max Drawdown", base, "max_drawdown"))
    lines.append(_metric_line("Win Rate", base, "win_rate"))
    lines.append(_metric_line("Loss Rate", base, "loss_rate"))
    lines.append(_metric_line("Profit Factor", base, "profit_factor"))
    lines.append(_metric_line("Sharpe Ratio", base, "sharpe_ratio"))
    lines.append(_metric_line("Total Trades", base, "total_trades"))
    lines.append(_metric_line("Timeouts", base, "timeouts"))
    lines.append("")
    lines.append("--- Meta-Model Best Threshold ---")
    lines.append(_metric_line("Threshold", {"value": wf_results.get("best_threshold", "n/a")}, "value"))
    lines.append(_metric_line("OOS Return", best, "total_return"))
    lines.append(_metric_line("Max Drawdown", best, "max_drawdown"))
    lines.append(_metric_line("Win Rate", best, "win_rate"))
    lines.append(_metric_line("Loss Rate", best, "loss_rate"))
    lines.append(_metric_line("Profit Factor", best, "profit_factor"))
    lines.append(_metric_line("Sharpe Ratio", best, "sharpe_ratio"))
    lines.append(_metric_line("Total Trades", best, "total_trades"))
    lines.append(_metric_line("Timeouts", best, "timeouts"))
    lines.append("")
    position_sizing = best.get("position_sizing", {})
    if position_sizing:
        lines.append("--- Position Sizing ---")
        lines.append(f"Vol-target constant used:  {position_sizing.get('target_vol_used', 'n/a')}")
        lines.append(_stat_line("Vol scale", position_sizing.get("vol_scale", {})))
        lines.append(_stat_line("Confidence scale", position_sizing.get("conf_scale", {})))
        lines.append(_stat_line("Combined scale", position_sizing.get("combined_scale", {})))
        lines.append("")
    adaptive_sl = best.get("adaptive_sl", {})
    if adaptive_sl:
        tightened = adaptive_sl.get("tightened", {})
        standard = adaptive_sl.get("standard", {})
        lines.append("--- Adaptive SL ---")
        lines.append(
            f"Tightened SL (<0.60):      trades={tightened.get('trades', 0)} | "
            f"WR={tightened.get('win_rate', 'n/a')} | PF={tightened.get('profit_factor', 'n/a')}"
        )
        lines.append(
            f"Standard SL (>=0.60):      trades={standard.get('trades', 0)} | "
            f"WR={standard.get('win_rate', 'n/a')} | PF={standard.get('profit_factor', 'n/a')}"
        )
        lines.append("")
    lines.append("--- Threshold Sweep ---")
    lines.append(f"{'Threshold':<10} {'Trades':>8} {'WinRate':>10} {'PF':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10}")
    lines.append("-" * 72)
    for result in threshold_results:
        lines.append(
            f"{result.get('threshold', 'n/a')!s:<10} "
            f"{int(result.get('total_trades', 0)):>8} "
            f"{str(result.get('win_rate', 'n/a')):>10} "
            f"{str(result.get('profit_factor', 'n/a')):>8} "
            f"{str(result.get('sharpe_ratio', 'n/a')):>8} "
            f"{str(result.get('total_return', 'n/a')):>10} "
            f"{str(result.get('max_drawdown', 'n/a')):>10}"
        )
    lines.append("")
    lines.append("--- Detailed Checks ---")
    for check, passed in wf_results.get("checks", {}).items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {check}")
    lines.append("")
    lines.append("--- Per-Fold Breakdown ---")
    for fold in wf_results.get("fold_results", []):
        if "warning" in fold:
            lines.append(f"  Fold {fold.get('fold', '?')}: SKIPPED | {fold['warning']}")
            continue
        lines.append(
            f"  Fold {fold.get('fold', '?')}: "
            f"train {fold.get('train_start', 'n/a')} → {fold.get('train_end', 'n/a')} | "
            f"test {fold.get('test_start', 'n/a')} → {fold.get('test_end', 'n/a')} | "
            f"TrainSignals={int(fold.get('train_signals', 0))} | "
            f"TestSignals={int(fold.get('test_signals', 0))} | "
            f"IS={float(fold.get('is_accuracy', 0.0)):.1%} | "
            f"OOS={float(fold.get('oos_accuracy', 0.0)):.1%} | "
            f"Overfit={float(fold.get('overfitting_ratio', 0.0)):.2f}"
        )
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"MODEL STATUS: {wf_results.get('readiness', 'NOT READY')}")
    lines.append("=" * 80)
    return "\n".join(lines)
