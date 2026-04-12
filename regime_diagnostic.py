"""
regime_diagnostic.py

Change summary:
- Adds a regime segmentation diagnostic for the raw base-signal layer before
  meta-label training.
- Buckets signals by volatility, trend strength, trend direction, session, and
  day of week to check whether any conditional edge exists.
- Flags regime buckets that meet minimum trade count and edge thresholds so the
  pipeline can decide whether meta-labeling is worth pursuing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DiagnosticConfig:
    atr_period: int = 20
    tp_atr_mult: float = 1.5
    sl_atr_mult: float = 1.0
    forward_horizon: int = 10

    vol_lookback: int = 20
    vol_quantiles: Tuple[float, ...] = (0.33, 0.66)
    trend_lookback: int = 50
    trend_quantiles: Tuple[float, ...] = (0.33, 0.66)

    sessions: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "Asia": (0, 8),
            "London": (8, 14),
            "NY": (14, 21),
            "Off": (21, 24),
        }
    )

    min_trades_per_bucket: int = 30
    edge_threshold_wr: float = 0.42
    edge_threshold_pf: float = 1.15


class RegimeDiagnostic:
    """
    Determine whether raw base signals show any exploitable conditional edge.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close",
        outcome_col: Optional[str] = None,
        config: Optional[DiagnosticConfig] = None,
    ):
        self.cfg = config or DiagnosticConfig()
        self.df = df.copy()
        self.signal_col = signal_col
        self.price_col = price_col
        self.outcome_col = outcome_col or "outcome"

        self._validate()
        self._compute_regime_features()
        if self.outcome_col not in self.df.columns:
            self._compute_outcomes()

    def _validate(self) -> None:
        required = [self.price_col, "high", "low", "open", self.signal_col]
        missing = [column for column in required if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

    def _atr(self) -> pd.Series:
        high = self.df["high"]
        low = self.df["low"]
        close = self.df[self.price_col]
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(self.cfg.atr_period, min_periods=self.cfg.atr_period).mean()

    def _compute_outcomes(self) -> None:
        atr = self._atr()
        signals = self.df[self.signal_col]
        outcomes = pd.Series(0, index=self.df.index, dtype=int)
        pnl = pd.Series(0.0, index=self.df.index, dtype=float)
        index_list = list(self.df.index)

        for timestamp in signals[signals != 0].index:
            signal_pos = index_list.index(timestamp)
            entry_pos = signal_pos + 1
            if entry_pos >= len(index_list):
                continue

            direction = int(signals.loc[timestamp])
            atr_value = float(atr.loc[timestamp]) if pd.notna(atr.loc[timestamp]) else np.nan
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue

            entry_price = float(self.df["open"].iloc[entry_pos])
            tp_dist = self.cfg.tp_atr_mult * atr_value
            sl_dist = self.cfg.sl_atr_mult * atr_value

            resolved = False
            scan_stop = min(entry_pos + self.cfg.forward_horizon, len(index_list))
            for row_pos in range(entry_pos, scan_stop):
                bar_high = float(self.df["high"].iloc[row_pos])
                bar_low = float(self.df["low"].iloc[row_pos])

                if direction == 1:
                    if bar_high >= entry_price + tp_dist:
                        outcomes.loc[timestamp] = 1
                        pnl.loc[timestamp] = tp_dist
                        resolved = True
                        break
                    if bar_low <= entry_price - sl_dist:
                        outcomes.loc[timestamp] = -1
                        pnl.loc[timestamp] = -sl_dist
                        resolved = True
                        break
                else:
                    if bar_low <= entry_price - tp_dist:
                        outcomes.loc[timestamp] = 1
                        pnl.loc[timestamp] = tp_dist
                        resolved = True
                        break
                    if bar_high >= entry_price + sl_dist:
                        outcomes.loc[timestamp] = -1
                        pnl.loc[timestamp] = -sl_dist
                        resolved = True
                        break

            if not resolved:
                exit_pos = min(entry_pos + self.cfg.forward_horizon - 1, len(index_list) - 1)
                exit_price = float(self.df[self.price_col].iloc[exit_pos])
                pnl.loc[timestamp] = float(direction * (exit_price - entry_price))
                outcomes.loc[timestamp] = 0

        self.df[self.outcome_col] = outcomes
        self.df["pnl"] = pnl

    def _compute_regime_features(self) -> None:
        close = self.df[self.price_col]
        log_returns = np.log(close / close.shift(1))
        self.df["realised_vol"] = log_returns.rolling(self.cfg.vol_lookback, min_periods=self.cfg.vol_lookback).std()

        sma = close.rolling(self.cfg.trend_lookback, min_periods=self.cfg.trend_lookback).mean()
        atr = self._atr()
        self.df["trend_strength"] = (sma - sma.shift(5)) / (atr + 1e-10)
        self.df["trend_dir"] = np.where(close > sma, "bullish", np.where(close < sma, "bearish", "flat"))

        utc_index = self.df.index.tz_convert("UTC") if self.df.index.tz is not None else self.df.index
        hours = utc_index.hour

        def _session(hour: int) -> str:
            for name, (start, end) in self.cfg.sessions.items():
                if start <= hour < end:
                    return name
            return "Off"

        self.df["session"] = [_session(int(hour)) for hour in hours]
        self.df["dow"] = self.df.index.day_name()

    def _quantile_bucket(self, column: str, quantiles: Tuple[float, ...]) -> pd.Series:
        values = self.df[column].dropna()
        if values.empty:
            return pd.Series("N/A", index=self.df.index)
        breaks = [float(values.quantile(q)) for q in quantiles]
        labels: list[str] = []
        for value in self.df[column]:
            if pd.isna(value):
                labels.append("N/A")
            elif value <= breaks[0]:
                labels.append("Low")
            elif value <= breaks[1]:
                labels.append("Mid")
            else:
                labels.append("High")
        return pd.Series(labels, index=self.df.index)

    @staticmethod
    def _bucket_stats(subset: pd.DataFrame, outcome_col: str) -> Dict[str, object]:
        trades = int(len(subset))
        wins = int((subset[outcome_col] == 1).sum())
        losses = int((subset[outcome_col] == -1).sum())
        timeouts = int((subset[outcome_col] == 0).sum())
        win_rate = wins / trades if trades else 0.0
        loss_rate = losses / trades if trades else 0.0

        if "pnl" in subset.columns:
            gross_profit = float(subset.loc[subset["pnl"] > 0, "pnl"].sum())
            gross_loss = abs(float(subset.loc[subset["pnl"] < 0, "pnl"].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            avg_pnl = float(subset["pnl"].mean())
        else:
            profit_factor = (wins * 1.5) / (losses * 1.0) if losses > 0 else float("inf")
            avg_pnl = np.nan

        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "win_rate": round(win_rate * 100, 1),
            "loss_rate": round(loss_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_pnl": round(avg_pnl, 6) if pd.notna(avg_pnl) else None,
        }

    def run(self) -> Dict[str, object]:
        self.df["vol_regime"] = self._quantile_bucket("realised_vol", self.cfg.vol_quantiles)
        self.df["trend_regime"] = self._quantile_bucket("trend_strength", self.cfg.trend_quantiles)
        signals = self.df[self.df[self.signal_col] != 0].copy()

        report: Dict[str, object] = {
            "overall": self._bucket_stats(signals, self.outcome_col),
            "segments": {},
            "conditional_edge_found": False,
            "edge_buckets": [],
        }

        dimensions = {
            "volatility_regime": "vol_regime",
            "trend_strength": "trend_regime",
            "trend_direction": "trend_dir",
            "session": "session",
            "day_of_week": "dow",
        }

        for dimension_name, column in dimensions.items():
            results: Dict[str, object] = {}
            for bucket_value, group in signals.groupby(column):
                stats = self._bucket_stats(group, self.outcome_col)
                results[str(bucket_value)] = stats
                if (
                    stats["trades"] >= self.cfg.min_trades_per_bucket
                    and (
                        stats["win_rate"] >= self.cfg.edge_threshold_wr * 100
                        or stats["profit_factor"] >= self.cfg.edge_threshold_pf
                    )
                ):
                    report["conditional_edge_found"] = True
                    report["edge_buckets"].append({"dimension": dimension_name, "bucket": bucket_value, **stats})
            report["segments"][dimension_name] = results

        cross_results: Dict[str, object] = {}
        for (vol_bucket, trend_bucket), group in signals.groupby(["vol_regime", "trend_regime"]):
            bucket_name = f"{vol_bucket}_vol x {trend_bucket}_trend"
            stats = self._bucket_stats(group, self.outcome_col)
            cross_results[bucket_name] = stats
            if (
                stats["trades"] >= self.cfg.min_trades_per_bucket
                and (
                    stats["win_rate"] >= self.cfg.edge_threshold_wr * 100
                    or stats["profit_factor"] >= self.cfg.edge_threshold_pf
                )
            ):
                report["conditional_edge_found"] = True
                report["edge_buckets"].append({"dimension": "vol x trend", "bucket": bucket_name, **stats})
        report["segments"]["vol_x_trend"] = cross_results
        return report

    def print_report(self, report: Dict[str, object]) -> None:
        print("=" * 80)
        print("REGIME SEGMENTATION DIAGNOSTIC")
        print("=" * 80)
        overall = report["overall"]
        print("\nOVERALL BASE SIGNALS")
        print(f"  Trades:        {overall['trades']}")
        print(f"  Win Rate:      {overall['win_rate']}%")
        print(f"  Loss Rate:     {overall['loss_rate']}%")
        print(f"  Profit Factor: {overall['profit_factor']}")
        if overall["avg_pnl"] is not None:
            print(f"  Avg PnL:       {overall['avg_pnl']}")

        for dimension_name, buckets in report["segments"].items():
            print("\n" + "-" * 80)
            print(f"  DIMENSION: {dimension_name.upper()}")
            print("-" * 80)
            print(f"  {'Bucket':<25} {'Trades':>7} {'WinRate':>8} {'LossRate':>9} {'PF':>7} {'AvgPnL':>10}")
            print(f"  {'-' * 70}")
            for bucket_value, stats in sorted(buckets.items(), key=lambda item: item[1]["win_rate"], reverse=True):
                avg_pnl = f"{stats['avg_pnl']:.6f}" if stats["avg_pnl"] is not None else "N/A"
                flag = ""
                if (
                    stats["trades"] >= self.cfg.min_trades_per_bucket
                    and (
                        stats["win_rate"] >= self.cfg.edge_threshold_wr * 100
                        or stats["profit_factor"] >= self.cfg.edge_threshold_pf
                    )
                ):
                    flag = "  <EDGE>"
                print(
                    f"  {str(bucket_value):<25} {stats['trades']:>7} {stats['win_rate']:>7.1f}% "
                    f"{stats['loss_rate']:>8.1f}% {stats['profit_factor']:>7.2f} {avg_pnl:>10}{flag}"
                )

        print("\n" + "=" * 80)
        if report["conditional_edge_found"]:
            print("CONDITIONAL EDGE DETECTED:")
            for bucket in report["edge_buckets"]:
                print(
                    f"  - [{bucket['dimension']}] {bucket['bucket']}: "
                    f"WR={bucket['win_rate']}%, PF={bucket['profit_factor']}, n={bucket['trades']}"
                )
            print("These regimes are candidates for the meta-label filter.")
        else:
            print("NO CONDITIONAL EDGE FOUND.")
            print("Fix the base signals first; meta-labeling has nothing to filter.")
        print("=" * 80)

    def plot(self, report: Dict[str, object], save_path: str = "regime_diagnostic.png") -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping regime diagnostic plot.")
            return

        dimensions = [dimension for dimension in report["segments"] if dimension != "vol_x_trend"]
        fig, axes = plt.subplots(1, len(dimensions) + 1, figsize=(5 * (len(dimensions) + 1), 5), squeeze=False)
        axes = axes[0]
        overall_wr = report["overall"]["win_rate"]

        for idx, dimension in enumerate(dimensions):
            axis = axes[idx]
            buckets = report["segments"][dimension]
            names = list(buckets.keys())
            win_rates = [buckets[name]["win_rate"] for name in names]
            trades = [buckets[name]["trades"] for name in names]
            colors = ["#2ecc71" if wr >= self.cfg.edge_threshold_wr * 100 else "#e74c3c" for wr in win_rates]
            bars = axis.bar(names, win_rates, color=colors, edgecolor="white", linewidth=0.5)
            axis.axhline(y=overall_wr, color="gray", linestyle="--", alpha=0.7)
            axis.axhline(y=self.cfg.edge_threshold_wr * 100, color="blue", linestyle=":", alpha=0.6)
            axis.set_title(dimension.replace("_", " ").title())
            axis.set_ylabel("Win Rate %")
            axis.tick_params(axis="x", rotation=45)
            for bar, count in zip(bars, trades, strict=False):
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"n={count}", ha="center", va="bottom", fontsize=7)

        cross_axis = axes[-1]
        cross = report["segments"].get("vol_x_trend", {})
        cross_names = list(cross.keys())
        cross_wrs = [cross[name]["win_rate"] for name in cross_names]
        cross_trades = [cross[name]["trades"] for name in cross_names]
        cross_colors = ["#2ecc71" if wr >= self.cfg.edge_threshold_wr * 100 else "#e74c3c" for wr in cross_wrs]
        bars = cross_axis.bar(range(len(cross_names)), cross_wrs, color=cross_colors, edgecolor="white", linewidth=0.5)
        cross_axis.set_xticks(range(len(cross_names)))
        cross_axis.set_xticklabels(cross_names, rotation=60, fontsize=6)
        cross_axis.axhline(y=overall_wr, color="gray", linestyle="--", alpha=0.7)
        cross_axis.set_title("Vol x Trend")
        cross_axis.set_ylabel("Win Rate %")
        for bar, count in zip(bars, cross_trades, strict=False):
            cross_axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"n={count}", ha="center", va="bottom", fontsize=6)

        fig.suptitle("Regime Segmentation Diagnostic", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
        plt.close(fig)
