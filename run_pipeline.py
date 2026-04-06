"""
run_pipeline.py

Main pipeline. Runs two evaluation modes and compares:

Mode A: Direct prediction (primary model predicts direction)
Mode B: Meta-labeling (base signals filtered by meta-model)
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import List

import pandas as pd

from features import feature_engineering_h1
from labelling import generate_directional_targets, regime_filter, verify_label_distribution
from meta_labeling import generate_base_signals, label_signal_outcomes
from report import generate_backtest_report
from resampler import resample_ohlcv
from validation import detect_lookahead_bias, walk_forward_meta_labeling, walk_forward_validation


def _parse_histdata_no_header(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip().str.zfill(6)
    parsed = pd.to_datetime(combined, format="%Y%m%d %H%M%S", errors="coerce")
    if parsed.isna().all():
        combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(combined, format="%Y.%m.%d %H:%M", errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"Could not parse HistData timestamps in {csv_path}")

    out = df.drop(columns=["date", "time"]).copy()
    out.index = parsed.dt.tz_localize("UTC")
    out = out.apply(pd.to_numeric, errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return out


def _parse_standard_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    dt_candidates = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "datetime", "timestamp"])]
    out = df.copy()
    if dt_candidates:
        dt_col = dt_candidates[0]
        out[dt_col] = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
        out = out.set_index(dt_col)
    elif df.columns[0] == "Unnamed: 0":
        out.iloc[:, 0] = pd.to_datetime(out.iloc[:, 0], utc=True, errors="coerce")
        out = out.set_index(out.columns[0])
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")

    col_map = {}
    for column in out.columns:
        lower = column.lower().strip()
        if "open" in lower and "open" not in col_map.values():
            col_map[column] = "open"
        elif "high" in lower and "high" not in col_map.values():
            col_map[column] = "high"
        elif "low" in lower and "low" not in col_map.values():
            col_map[column] = "low"
        elif "close" in lower and "close" not in col_map.values():
            col_map[column] = "close"
        elif "vol" in lower and "volume" not in col_map.values():
            col_map[column] = "volume"
    out = out.rename(columns=col_map)

    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"]).sort_index()
    return out


def load_data(data_dir: str = ".") -> pd.DataFrame:
    """
    Load EURUSD OHLCV data from CSV and auto-detect the layout.
    """
    base_dir = Path(data_dir).resolve()
    raw_histdata_files = sorted(
        str(path)
        for path in (base_dir / "data" / "raw").rglob("*.csv")
        if "EURUSD" in path.name.upper()
    )
    if raw_histdata_files:
        print(f"Loading {len(raw_histdata_files)} raw HistData files from {base_dir / 'data' / 'raw'}")
        frames = [_parse_histdata_no_header(path) for path in raw_histdata_files]
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()
        print(f"Detected bar interval: {median_diff}")
        print(f"Total bars loaded: {len(df)}")
        print(f"Date range: {df.index.min()} → {df.index.max()}")
        return df

    patterns = [
        str(base_dir / "*.csv"),
        str(base_dir / "data" / "*.csv"),
    ]
    csv_files: list[str] = []
    for pattern in patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))

    csv_files = sorted(
        {
            path
            for path in csv_files
            if "EURUSD" in Path(path).name.upper() and "FEATURES" not in Path(path).name.upper()
        }
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {base_dir}. Place your EURUSD OHLCV CSV there."
        )

    csv_path = max(csv_files, key=os.path.getsize)
    print(f"Loading: {csv_path}")

    try:
        df = _parse_histdata_no_header(csv_path)
    except Exception:
        df = _parse_standard_csv(csv_path)

    diffs = df.index.to_series().diff().dropna()
    median_diff = diffs.median()
    print(f"Detected bar interval: {median_diff}")
    print(f"Total bars loaded: {len(df)}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    return df


def run_mode_a_direct_prediction(df_h1: pd.DataFrame, feature_cols: List[str], label_col: str) -> dict:
    print("\n" + "=" * 60)
    print("MODE A: DIRECT PREDICTION (trade-return targets)")
    print("=" * 60)
    return walk_forward_validation(
        df_h1,
        feature_cols,
        label_col=label_col,
        train_months=6,
        test_months=1,
        purge_bars=24,
        embargo_bars=12,
        risk_per_trade=0.01,
        tp_atr_mult=1.5,
        sl_atr_mult=1.0,
    )


def run_mode_b_meta_labeling(df_h1: pd.DataFrame, feature_cols: List[str]) -> dict | None:
    print("\n" + "=" * 60)
    print("MODE B: META-LABELING (base signal + meta filter)")
    print("=" * 60)

    signals = generate_base_signals(df_h1)
    signal_count = int((signals != 0).sum())
    print(f"Base signals generated: {signal_count} ({signal_count / len(df_h1) * 100:.1f}% of bars)")

    outcomes = label_signal_outcomes(
        df_h1,
        signals,
        tp_atr_mult=1.5,
        sl_atr_mult=1.0,
        max_holding_bars=24,
    )
    valid = outcomes.dropna()
    if len(valid) == 0:
        print("WARNING: No valid signal outcomes. Check data.")
        return None

    print(f"Base strategy win rate (no filter): {valid.mean():.1%}")
    return walk_forward_meta_labeling(
        df_h1,
        feature_cols,
        signals,
        outcomes,
        train_months=6,
        test_months=1,
        purge_bars=24,
        embargo_bars=12,
        risk_per_trade=0.01,
        confidence_threshold=0.55,
    )


def _parse_checks_passed(value: object) -> int:
    if isinstance(value, str) and "/" in value:
        try:
            return int(value.split("/", 1)[0])
        except ValueError:
            return 0
    if isinstance(value, int):
        return value
    return 0


def main() -> None:
    start_time = time.time()

    print("STEP 1: Loading data...")
    raw = load_data(".")

    print("\nSTEP 2: Resampling to H1...")
    diffs = raw.index.to_series().diff().dropna()
    if not diffs.empty and diffs.median() >= pd.Timedelta(hours=1):
        df_h1_raw = raw.copy()
    else:
        df_h1_raw = resample_ohlcv(raw, target_tf="1h")
    print(f"H1 bars: {len(df_h1_raw)}")
    if len(df_h1_raw) < 2000:
        print("WARNING: < 2000 H1 bars. Need 2+ years for reliable walk-forward validation.")

    print("\nSTEP 3: Engineering features (causally clean)...")
    df_h1 = feature_engineering_h1(df_h1_raw)
    print(f"Features computed. Bars after warmup: {len(df_h1)}")

    df_h1["regime"] = regime_filter(df_h1_raw).reindex(df_h1.index).fillna(0).astype(int)
    regime_dist = verify_label_distribution(df_h1["regime"])
    print(f"Regime feature distribution: {regime_dist}")
    if not regime_dist["healthy"]:
        print(f"WARNING: Unhealthy regime distribution: {regime_dist['diagnosis']}")

    feature_cols = [
        column
        for column in df_h1.columns
        if column not in ["open", "high", "low", "close", "volume", "target_long", "target_short", "target_best"]
    ]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:8]}{'...' if len(feature_cols) > 8 else ''}")

    print("\nSTEP 4: Generating trade-return targets...")
    targets = generate_directional_targets(
        df_h1,
        tp_atr_mult=1.5,
        sl_atr_mult=1.0,
        max_holding_bars=24,
    )
    df_h1 = pd.concat([df_h1, targets], axis=1)
    df_h1 = df_h1.dropna(subset=["target_best"]).copy()
    print(f"Bars with valid targets: {len(df_h1)}")

    dist = df_h1["target_best"].value_counts(normalize=True)
    print(
        "Target distribution:\n"
        f"  +1 (long wins):  {dist.get(1.0, 0):.1%}\n"
        f"  -1 (short wins): {dist.get(-1.0, 0):.1%}\n"
        f"   0 (ambiguous):  {dist.get(0.0, 0):.1%}"
    )

    print("\nSTEP 5: Lookahead bias check...")
    bias_warnings = detect_lookahead_bias(df_h1, feature_cols, "target_best")
    for warning in bias_warnings:
        print(f"  {warning}")

    print("\nCorrelation audit (|corr| > 0.15 with target):")
    for column in feature_cols:
        corr = df_h1[column].corr(df_h1["target_best"])
        if pd.notna(corr) and abs(float(corr)) > 0.15:
            flag = " ⚠️ INVESTIGATE" if abs(float(corr)) > 0.30 else ""
            print(f"  {column}: {float(corr):.4f}{flag}")

    results_a = run_mode_a_direct_prediction(df_h1, feature_cols, "target_best")
    report_a = generate_backtest_report(results_a, title="MODE A")
    print(report_a)

    results_b = run_mode_b_meta_labeling(df_h1, feature_cols)
    report_b = None
    if results_b is not None:
        report_b = generate_backtest_report(results_b, title="MODE B")
        print(report_b)

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Mode A':>12} {'Mode B':>12}")
    print("-" * 49)
    for metric in ["total_return", "max_drawdown", "win_rate", "profit_factor", "sharpe_ratio", "total_trades"]:
        value_a = results_a.get(metric, "N/A")
        value_b = results_b.get(metric, "N/A") if results_b is not None else "N/A"
        print(f"{metric:<25} {str(value_a):>12} {str(value_b):>12}")

    best_mode = "A"
    best_results = results_a
    if results_b is not None and _parse_checks_passed(results_b.get("checks_passed")) > _parse_checks_passed(results_a.get("checks_passed")):
        best_mode = "B"
        best_results = results_b

    print(f"\nBest performing mode: {best_mode}")
    print(f"Verdict: {best_results['verdict']}")

    report_sections = [report_a]
    if report_b is not None:
        report_sections.append(report_b)
    full_report = "\n\n".join(report_sections)
    report_path = Path("backtest_report.txt")
    report_path.write_text(full_report, encoding="utf-8")
    print("\nReport saved to backtest_report.txt")

    elapsed = time.time() - start_time
    print(f"\nTotal pipeline time: {elapsed:.1f}s")

    try:
        win_rate = float(str(best_results["win_rate"]).strip("%")) / 100.0
        sharpe_ratio = float(best_results["sharpe_ratio"])
        if win_rate > 0.65:
            print("\nWARNING: Win rate > 65% is suspicious. Possible residual leakage.")
        if sharpe_ratio > 2.0:
            print("WARNING: Sharpe > 2.0 is suspicious. Possible residual leakage.")
        if win_rate > 0.65 and sharpe_ratio > 2.0:
            print("LIKELY STILL LEAKING. Re-audit features, targets, and fold separation.")
    except (KeyError, TypeError, ValueError):
        pass


if __name__ == "__main__":
    main()
