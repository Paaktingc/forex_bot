"""
report.py

Formatting helpers for causally clean walk-forward backtest reports.
"""

from __future__ import annotations


def generate_backtest_report(wf_results: dict, title: str = "BACKTEST") -> str:
    """
    Pretty-print a walk-forward report with causality metadata.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"VERDICT: {wf_results.get('verdict', 'UNKNOWN')}")
    lines.append(f"Checks Passed: {wf_results.get('checks_passed', '0/8')}")
    if wf_results.get("reason"):
        lines.append(f"Reason: {wf_results['reason']}")
    lines.append("")
    lines.append("--- Causality Status ---")
    lines.append(f"Target type:               {wf_results.get('target_type', 'trade-return')}")
    lines.append(f"Purge gap:                 {wf_results.get('purge_bars', 'n/a')} bars")
    lines.append(f"Embargo gap:               {wf_results.get('embargo_bars', 'n/a')} bars")
    max_corr = wf_results.get("max_feature_target_corr", "n/a")
    if isinstance(max_corr, (float, int)):
        max_corr = f"{float(max_corr):.3f}"
    lines.append(f"Feature-target max corr:   {max_corr}")
    lines.append(f"Leakage risk:              {wf_results.get('leakage_risk', 'UNKNOWN')}")
    lines.append("")
    lines.append("--- Performance Summary ---")
    lines.append(f"Total Return (OOS):        {wf_results.get('total_return', 'n/a')}")
    lines.append(f"Max Drawdown (OOS):        {wf_results.get('max_drawdown', 'n/a')}")
    lines.append(f"Win Rate:                  {wf_results.get('win_rate', 'n/a')}")
    lines.append(f"Profit Factor:             {wf_results.get('profit_factor', 'n/a')}")
    lines.append(f"Sharpe Ratio (ann.):       {wf_results.get('sharpe_ratio', 'n/a')}")
    lines.append(f"Total OOS Trades:          {wf_results.get('total_trades', 'n/a')}")
    lines.append("")
    lines.append("--- Overfitting Analysis ---")
    lines.append(f"Avg IS Accuracy:           {wf_results.get('avg_is_accuracy', 'n/a')}")
    lines.append(f"Avg OOS Accuracy:          {wf_results.get('avg_oos_accuracy', 'n/a')}")
    lines.append(f"Overfit Ratio (IS/OOS):    {wf_results.get('overfit_ratio', 'n/a')}")
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
            f"IS={float(fold.get('is_accuracy', 0.0)):.1%} | "
            f"OOS={float(fold.get('oos_accuracy', 0.0)):.1%} | "
            f"Trades={int(fold.get('oos_trades', 0))} | "
            f"Wins={int(fold.get('oos_wins', 0))} | "
            f"Overfit={float(fold.get('overfitting_ratio', 0.0)):.2f}"
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"THE5ERS BOOTCAMP READINESS: {wf_results.get('verdict', 'UNKNOWN')}")
    lines.append("=" * 72)
    return "\n".join(lines)
