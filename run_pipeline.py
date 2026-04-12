"""
run_pipeline.py

Change summary:
- Switched the research pipeline from bar-wise directional prediction to
  meta-labeling on raw rule-based candidate signals.
- The run now evaluates raw base signals and meta-filtered signals across
  multiple probability thresholds.
"""

from __future__ import annotations

import io
import glob
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from features import META_FEATURE_COLS, feature_engineering_h1
from feature_importance import FeatureImportancePipeline, PipelineConfig
from labelling import generate_base_strategy_signals, label_base_signals_with_triple_barrier
from regime_diagnostic import DiagnosticConfig, RegimeDiagnostic
from report import generate_backtest_report
from resampler import resample_ohlcv
from validation import detect_lookahead_bias, walk_forward_meta_labeling

PURGE_GAP = 10
EMBARGO_GAP = 20
THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
META_CONFIG = {
    "tp_atr_mult": 1.5,
    "forward_horizon": 10,
    "target_vol": None,
    "vol_scale_clamp": (0.25, 2.0),
    "conf_scale_range": (0.5, 2.0),
    "combined_scale_clamp": (0.2, 3.0),
    "adaptive_sl_threshold": 0.60,
    "tight_sl_mult": 0.8,
    "standard_sl_mult": 1.0,
}


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
        print(f"Detected bar interval: {diffs.median()}")
        print(f"Total bars loaded: {len(df)}")
        print(f"Date range: {df.index.min()} → {df.index.max()}")
        return df

    patterns = [str(base_dir / "*.csv"), str(base_dir / "data" / "*.csv")]
    csv_files: list[str] = []
    for pattern in patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))

    csv_files = sorted({path for path in csv_files if "EURUSD" in Path(path).name.upper()})
    if not csv_files:
        raise FileNotFoundError(f"No EURUSD CSV files found under {base_dir}")

    csv_path = max(csv_files, key=os.path.getsize)
    print(f"Loading: {csv_path}")
    try:
        df = _parse_histdata_no_header(csv_path)
    except Exception:
        df = _parse_standard_csv(csv_path)
    diffs = df.index.to_series().diff().dropna()
    print(f"Detected bar interval: {diffs.median()}")
    print(f"Total bars loaded: {len(df)}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    return df


def _attach_signal_metadata(df_h1: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    out = df_h1.copy()
    out["base_signal"] = 0
    out["direction_code"] = 0
    out["reason_code"] = ""
    out["reason_sma_cross"] = 0
    out["reason_breakout"] = 0
    out["barrier_label"] = pd.NA
    out["outcome_binary"] = pd.NA
    out["realized_r"] = pd.NA

    if signals.empty:
        return out

    aligned_index = out.index.intersection(signals.index)
    signal_slice = signals.loc[aligned_index]
    out.loc[aligned_index, "base_signal"] = signal_slice["signal"].astype(int)
    out.loc[aligned_index, "direction_code"] = signal_slice["signal"].astype(int)
    out.loc[aligned_index, "reason_code"] = signal_slice["reason_code"].astype(str)
    out.loc[aligned_index, "reason_sma_cross"] = signal_slice["reason_code"].str.contains("sma_cross").astype(int)
    out.loc[aligned_index, "reason_breakout"] = signal_slice["reason_code"].str.contains("breakout").astype(int)
    out.loc[aligned_index, "barrier_label"] = signal_slice["barrier_label"].astype(int)
    out.loc[aligned_index, "outcome_binary"] = signal_slice["outcome_binary"]
    out.loc[aligned_index, "realized_r"] = signal_slice["realized_r"].astype(float)
    return out


def _capture_output(func, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    output = buffer.getvalue().strip()
    if output:
        print(output)
    return output


def _build_augmented_feature_frame(df_meta: pd.DataFrame, diagnostic_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    augmented = df_meta.copy()
    aligned = diagnostic_frame.reindex(augmented.index)

    regime_numeric = ["realised_vol", "trend_strength"]
    regime_categorical = ["vol_regime", "trend_regime", "trend_dir", "session", "dow"]

    for column in regime_numeric:
        augmented[column] = aligned[column]

    encoded = pd.get_dummies(
        aligned[regime_categorical].fillna("N/A"),
        prefix=["regime_vol", "regime_trend", "regime_dir", "regime_session", "regime_dow"],
        dtype=int,
    )
    augmented = augmented.join(encoded)

    candidate_cols = list(
        dict.fromkeys(
            list(META_FEATURE_COLS)
            + ["direction_code", "reason_sma_cross", "reason_breakout"]
            + regime_numeric
            + list(encoded.columns)
        )
    )
    return augmented, candidate_cols


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

    print("\nSTEP 3: Engineering signal-time features...")
    df_h1 = feature_engineering_h1(df_h1_raw)
    print(f"Feature rows after warmup: {len(df_h1)}")

    print("\nSTEP 4: Generating raw base signals...")
    signals = generate_base_strategy_signals(df_h1)
    print(f"Candidate signals: {len(signals)}")
    if not signals.empty:
        print(signals["reason_code"].value_counts().to_string())

    print("\nSTEP 5: Labeling signals with Triple Barrier...")
    labelled_signals = label_base_signals_with_triple_barrier(df_h1, signals)
    print(f"Signals with barrier labels: {len(labelled_signals)}")
    if not labelled_signals.empty:
        label_counts = labelled_signals["barrier_label"].value_counts(dropna=False).sort_index()
        print("Barrier outcome distribution:")
        for label_value, count in label_counts.items():
            print(f"  {label_value}: {count}")

    print("\nSTEP 6: Building meta-label dataset...")
    df_meta = _attach_signal_metadata(df_h1, labelled_signals)
    signal_rows = int((df_meta["base_signal"] != 0).sum())
    non_timeout_rows = int(((df_meta["base_signal"] != 0) & df_meta["outcome_binary"].notna()).sum())
    print(f"Signal rows available: {signal_rows}")
    print(f"Training rows after filtering TIMEOUT: {non_timeout_rows}")

    print("\nSTEP 7: Running regime diagnostic...")
    df_diag = df_meta.copy()
    df_diag["signal"] = df_diag["base_signal"].astype(int)
    df_diag["outcome"] = pd.to_numeric(df_diag["barrier_label"], errors="coerce").fillna(0).astype(int)
    regime_diag = RegimeDiagnostic(
        df_diag,
        signal_col="signal",
        price_col="close",
        outcome_col="outcome",
        config=DiagnosticConfig(),
    )
    regime_results = regime_diag.run()
    regime_report_text = _capture_output(regime_diag.print_report, regime_results)
    regime_diag.plot(regime_results, save_path="regime_diagnostic.png")

    if not regime_results.get("conditional_edge_found", False):
        final_lines = [
            regime_report_text,
            "",
            "PIPELINE STATUS",
            "No conditional edge was found in the raw base signals.",
            "Skipping permutation importance and meta-labeling training.",
            "Fix the entry logic before continuing.",
        ]
        final_report = "\n".join(line for line in final_lines if line is not None)
        Path("backtest_report.txt").write_text(final_report, encoding="utf-8")
        print("\nReport saved to backtest_report.txt")
        print(f"\nTotal pipeline time: {time.time() - start_time:.1f}s")
        return

    print("\nSTEP 8: Building augmented diagnostic feature set...")
    df_model, candidate_feature_cols = _build_augmented_feature_frame(df_meta, regime_diag.df)
    importance_mask = (df_model["base_signal"] != 0) & df_model["barrier_label"].notna()
    importance_mask &= df_model[candidate_feature_cols].notna().all(axis=1)
    X_importance = df_model.loc[importance_mask, candidate_feature_cols]
    y_importance = (df_model.loc[importance_mask, "barrier_label"] == 1).astype(int)
    print(f"Candidate feature columns: {len(candidate_feature_cols)}")
    print(f"Rows for feature importance: {len(X_importance)}")

    print("\nSTEP 9: Running permutation feature importance...")
    importance_pipe = FeatureImportancePipeline(
        X_importance,
        y_importance,
        config=PipelineConfig(
            n_folds=5,
            purge_bars=PURGE_GAP,
            embargo_bars=EMBARGO_GAP,
            max_features_to_keep=8,
        ),
    )
    importance_results = importance_pipe.run()
    importance_report_text = _capture_output(importance_pipe.print_report, importance_results)
    importance_pipe.plot(importance_results, save_path="feature_importance.png")

    selected_feature_cols = [item["feature"] for item in importance_results["feature_selection"]["selected"]]
    if not selected_feature_cols:
        selected_feature_cols = list(META_FEATURE_COLS) + ["direction_code", "reason_sma_cross", "reason_breakout"]
        print("No features beat the noise baseline. Falling back to the core meta feature set.")
    else:
        print(f"Selected feature columns ({len(selected_feature_cols)}):")
        for column in selected_feature_cols:
            print(f"  - {column}")

    print("\nSTEP 10: Lookahead bias check...")
    bias_frame = df_model.loc[df_model["outcome_binary"].notna()].copy()
    for warning in detect_lookahead_bias(bias_frame, selected_feature_cols, "outcome_binary"):
        print(f"  {warning}")

    print("\nCorrelation audit (|corr| > 0.15 with outcome_binary):")
    for column in selected_feature_cols:
        corr = bias_frame[column].corr(bias_frame["outcome_binary"])
        if pd.notna(corr) and abs(float(corr)) > 0.15:
            flag = " ⚠️ INVESTIGATE" if abs(float(corr)) > 0.30 else ""
            print(f"  {column}: {float(corr):.4f}{flag}")

    print("\nSTEP 11: Running meta-labeling walk-forward validation...")
    results = walk_forward_meta_labeling(
        df_model,
        selected_feature_cols,
        train_months=6,
        test_months=1,
        purge_gap=PURGE_GAP,
        embargo_gap=EMBARGO_GAP,
        risk_per_trade=0.01,
        thresholds=THRESHOLDS,
        meta_config=META_CONFIG,
    )
    results["regime_diagnostic"] = {
        "conditional_edge_found": bool(regime_results.get("conditional_edge_found", False)),
        "edge_bucket_count": len(regime_results.get("edge_buckets", [])),
        "edge_buckets": regime_results.get("edge_buckets", [])[:10],
    }
    results["feature_importance"] = {
        "candidate_feature_count": len(candidate_feature_cols),
        "selected_feature_count": len(selected_feature_cols),
        "selected_features": selected_feature_cols,
        "noise_threshold": importance_results["feature_selection"]["threshold"],
        "noise_median": importance_results["feature_selection"]["noise_median"],
        "shuffle_accuracy": importance_results["label_shuffle"]["mean_shuffle_acc"],
        "shuffle_accuracy_legacy": importance_results["label_shuffle"]["old_mean_shuffle_acc"],
        "shuffle_accuracy_balanced": importance_results["label_shuffle"]["balanced_mean_shuffle_acc"],
        "leakage_suspected": importance_results["label_shuffle"]["leakage_suspected"],
    }

    print("\nSTEP 12: Generating report...")
    report = generate_backtest_report(results)
    print(report)
    full_report = "\n\n".join(section for section in [regime_report_text, importance_report_text, report] if section)
    Path("backtest_report.txt").write_text(full_report, encoding="utf-8")
    print("\nReport saved to backtest_report.txt")
    print(f"\nTotal pipeline time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
