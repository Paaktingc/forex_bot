"""
validation.py

Change summary:
- Added volatility-targeted and confidence-weighted position sizing for the
  filtered meta-label trades.
- Re-simulates filtered trades with adaptive stop-loss logic based on model
  confidence while keeping the purged walk-forward CV scheme unchanged.
- Updates the backtest gates to the current H1 meta-labeling targets and
  exposes sizing / adaptive-SL diagnostics for reporting.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from meta_labeling import train_meta_model


def _fit_probability_calibrator(kind: str, train_prob: np.ndarray, y_train: np.ndarray):
    """
    Fit an in-sample probability calibrator on training predictions only.
    Returns a callable mapping raw probabilities -> calibrated probabilities.
    'none' is the identity (default; preserves legacy behaviour).
    """
    if kind == "none":
        return lambda p: p
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_prob, y_train)
        return lambda p: iso.predict(p)
    if kind == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(train_prob.reshape(-1, 1), y_train)
        return lambda p: lr.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown probability_calibration: {kind}")


def _position_confidence_multiplier(
    mode: str,
    probability: float,
    threshold: float,
    conf_scale_range: tuple,
    *,
    train_prob_cal: np.ndarray,
    train_r: np.ndarray,
) -> float:
    """
    Confidence-based position multiplier (pre vol-scale). 'confidence' is the
    legacy linear ramp (default). 'ecdf' sizes by the probability's percentile
    rank within training; 'kelly' uses a fractional-Kelly edge estimate.
    """
    lo, hi = float(conf_scale_range[0]), float(conf_scale_range[1])
    if mode == "confidence":
        m = 0.5 + (probability - threshold) / max(1e-9, 1.0 - threshold) * 1.5
    elif mode == "ecdf":
        rank = float(np.mean(train_prob_cal <= probability))
        m = 0.5 + rank * 1.5
    elif mode == "kelly":
        wins = train_r[train_r > 0]
        losses = train_r[train_r < 0]
        b = (wins.mean() / abs(losses.mean())) if (len(wins) and len(losses)) else 1.0
        b = max(b, 0.1)
        f = (probability * b - (1.0 - probability)) / b
        m = max(f, 0.0) * 4.0  # ~quarter-Kelly scaled into the multiplier band
    else:
        raise ValueError(f"Unknown sizing_mode: {mode}")
    return _clamp(m, lo, hi)


TARGET_DEFINITION = (
    "Signal observed after H1 bar t closes, entry at open[t+1], "
    "TP=+1.5 ATR(20)[t], SL adapts between 0.8/1.0 ATR(20)[t] by confidence, "
    "forward horizon=10 bars."
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _ensure_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    extra_columns: Iterable[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Validation requires a DatetimeIndex.")

    required = ["open", "high", "low", "close", *feature_cols, *extra_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required validation columns: {missing}")
    return df.sort_index().copy()


def _compute_feature_target_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[float, dict[str, float]]:
    correlations: dict[str, float] = {}
    max_corr = 0.0
    if label_col not in df.columns:
        return max_corr, correlations

    for column in feature_cols:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        corr = df[column].corr(df[label_col])
        if pd.isna(corr):
            continue
        corr_value = float(corr)
        correlations[column] = corr_value
        max_corr = max(max_corr, abs(corr_value))
    return max_corr, correlations


def _leakage_risk(max_corr: float) -> str:
    if max_corr > 0.30:
        return "HIGH"
    if max_corr > 0.15:
        return "MEDIUM"
    return "LOW"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_ratio(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "inf"


def _iter_fold_boundaries(
    index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    purge_gap: int,
    embargo_gap: int,
) -> list[dict[str, object]]:
    if len(index) == 0:
        return []

    folds: list[dict[str, object]] = []
    start_date = index.min()
    end_date = index.max()
    split_date = start_date + pd.DateOffset(months=train_months)
    fold_num = 0

    while split_date < end_date:
        fold_num += 1
        split_pos = int(index.searchsorted(split_date, side="left"))
        train_stop = max(0, split_pos - purge_gap)
        test_start = split_pos + embargo_gap
        test_end_date = split_date + pd.DateOffset(months=test_months)
        test_end = int(index.searchsorted(test_end_date, side="left"))

        if train_stop <= 0:
            split_date += pd.DateOffset(months=test_months)
            continue
        if test_start >= len(index):
            break
        if test_end <= test_start:
            split_date += pd.DateOffset(months=test_months)
            continue

        folds.append(
            {
                "fold": fold_num,
                "train_slice": slice(0, train_stop),
                "test_slice": slice(test_start, test_end),
                "train_start": index[0],
                "train_end": index[train_stop - 1],
                "test_start": index[test_start],
                "test_end": index[test_end - 1],
            }
        )
        split_date += pd.DateOffset(months=test_months)

    return folds


def _stat_block(values: pd.Series) -> dict[str, str]:
    if values.empty:
        return {"mean": "n/a", "median": "n/a", "std": "n/a"}
    return {
        "mean": f"{float(values.mean()):.3f}",
        "median": f"{float(values.median()):.3f}",
        "std": f"{float(values.std(ddof=0)):.3f}",
    }


def _group_stats(trades_df: pd.DataFrame, group_name: str) -> dict[str, object]:
    if trades_df.empty:
        return {"trades": 0, "win_rate": "0.0%", "profit_factor": "0.00"}
    wins = int((trades_df["barrier_label"] == 1).sum())
    total = int(len(trades_df))
    win_rate = wins / total if total else 0.0
    gross_profit = float(trades_df.loc[trades_df["pnl_value"] > 0, "pnl_value"].sum())
    gross_loss = abs(float(trades_df.loc[trades_df["pnl_value"] < 0, "pnl_value"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "group": group_name,
        "trades": total,
        "win_rate": f"{win_rate * 100:.1f}%",
        "profit_factor": _format_ratio(profit_factor),
    }


def _empty_metrics() -> dict[str, object]:
    return {
        "total_return_value": 0.0,
        "max_drawdown_value": 0.0,
        "win_rate_value": 0.0,
        "loss_rate_value": 0.0,
        "profit_factor_value": 0.0,
        "sharpe_ratio_value": 0.0,
        "total_trades": 0,
        "timeouts": 0,
        "total_return": "0.00%",
        "max_drawdown": "0.00%",
        "win_rate": "0.0%",
        "loss_rate": "0.0%",
        "profit_factor": "0.00",
        "sharpe_ratio": "0.00",
        "position_sizing": {
            "target_vol_used": "n/a",
            "vol_scale": {"mean": "n/a", "median": "n/a", "std": "n/a"},
            "conf_scale": {"mean": "n/a", "median": "n/a", "std": "n/a"},
            "combined_scale": {"mean": "n/a", "median": "n/a", "std": "n/a"},
        },
        "adaptive_sl": {
            "tightened": {"trades": 0, "win_rate": "0.0%", "profit_factor": "0.00"},
            "standard": {"trades": 0, "win_rate": "0.0%", "profit_factor": "0.00"},
        },
    }


def _summarise_trades(trades: list[dict[str, object]]) -> dict[str, object]:
    if not trades:
        return _empty_metrics()

    trades_df = pd.DataFrame(trades).sort_values("datetime")
    equity = (1.0 + trades_df["pct_return"]).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0

    total_return_value = float(equity.iloc[-1] - 1.0)
    max_drawdown_value = float(drawdown.min())
    total_trades = int(len(trades_df))
    wins = int((trades_df["barrier_label"] == 1).sum())
    losses = int((trades_df["barrier_label"] == -1).sum())
    timeouts = int((trades_df["barrier_label"] == 0).sum())

    win_rate_value = wins / total_trades if total_trades else 0.0
    loss_rate_value = losses / total_trades if total_trades else 0.0
    gross_profit = float(trades_df.loc[trades_df["pnl_value"] > 0, "pnl_value"].sum())
    gross_loss = abs(float(trades_df.loc[trades_df["pnl_value"] < 0, "pnl_value"].sum()))
    profit_factor_value = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_returns = (1.0 + trades_df.set_index("datetime")["pct_return"]).resample("1D").prod() - 1.0
    daily_returns = daily_returns[daily_returns != 0]
    if len(daily_returns) > 1 and float(daily_returns.std()) > 0:
        sharpe_ratio_value = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    else:
        sharpe_ratio_value = 0.0

    position_sizing = {
        "target_vol_used": "n/a",
        "vol_scale": {"mean": "n/a", "median": "n/a", "std": "n/a"},
        "conf_scale": {"mean": "n/a", "median": "n/a", "std": "n/a"},
        "combined_scale": {"mean": "n/a", "median": "n/a", "std": "n/a"},
    }
    if {"vol_scale", "conf_scale", "position_scale"}.issubset(trades_df.columns):
        target_vols = pd.to_numeric(trades_df.get("target_vol"), errors="coerce").dropna()
        if not target_vols.empty:
            position_sizing["target_vol_used"] = (
                f"mean={float(target_vols.mean()):.6f}, median={float(target_vols.median()):.6f}"
            )
        position_sizing["vol_scale"] = _stat_block(pd.to_numeric(trades_df["vol_scale"], errors="coerce").dropna())
        position_sizing["conf_scale"] = _stat_block(pd.to_numeric(trades_df["conf_scale"], errors="coerce").dropna())
        position_sizing["combined_scale"] = _stat_block(
            pd.to_numeric(trades_df["position_scale"], errors="coerce").dropna()
        )

    adaptive_sl = {
        "tightened": {"trades": 0, "win_rate": "0.0%", "profit_factor": "0.00"},
        "standard": {"trades": 0, "win_rate": "0.0%", "profit_factor": "0.00"},
    }
    if "sl_group" in trades_df.columns:
        adaptive_sl["tightened"] = _group_stats(trades_df[trades_df["sl_group"] == "tightened"], "tightened")
        adaptive_sl["standard"] = _group_stats(trades_df[trades_df["sl_group"] == "standard"], "standard")

    return {
        "total_return_value": total_return_value,
        "max_drawdown_value": max_drawdown_value,
        "win_rate_value": win_rate_value,
        "loss_rate_value": loss_rate_value,
        "profit_factor_value": profit_factor_value,
        "sharpe_ratio_value": sharpe_ratio_value,
        "total_trades": total_trades,
        "timeouts": timeouts,
        "total_return": _format_pct(total_return_value),
        "max_drawdown": _format_pct(max_drawdown_value),
        "win_rate": f"{win_rate_value * 100:.1f}%",
        "loss_rate": f"{loss_rate_value * 100:.1f}%",
        "profit_factor": _format_ratio(profit_factor_value),
        "sharpe_ratio": _format_ratio(sharpe_ratio_value),
        "position_sizing": position_sizing,
        "adaptive_sl": adaptive_sl,
    }


def _score_threshold_result(metrics: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(metrics["sharpe_ratio_value"]),
        float(metrics["profit_factor_value"]),
        float(metrics["total_return_value"]),
        int(metrics["total_trades"]),
    )


def _build_top_level_verdict(best_metrics: dict[str, object], overfit_ratio: float) -> tuple[str, str, dict[str, bool]]:
    checks = {
        "return_target": float(best_metrics["total_return_value"]) > 0.10,
        "max_dd_ok": abs(float(best_metrics["max_drawdown_value"])) < 0.15,
        "win_rate_ok": float(best_metrics["win_rate_value"]) > 0.40,
        "profit_factor_ok": float(best_metrics["profit_factor_value"]) > 1.20,
        "sharpe_ok": float(best_metrics["sharpe_ratio_value"]) > 0.50,
        "enough_trades": int(best_metrics["total_trades"]) > 50,
        "not_overfit": overfit_ratio < 1.5,
    }
    passed = int(sum(checks.values()))
    if passed >= 6:
        verdict = "GO"
    elif passed >= 4:
        verdict = "CONDITIONAL"
    else:
        verdict = "NO-GO"

    readiness = "READY" if verdict == "GO" else "NOT READY"
    return verdict, readiness, checks


def _prepare_realised_vol(clean: pd.DataFrame) -> pd.DataFrame:
    out = clean.copy()
    if "realised_vol_20" in out.columns:
        return out
    if "realised_vol" in out.columns:
        out["realised_vol_20"] = out["realised_vol"]
        return out

    log_returns = np.log(out["close"] / out["close"].shift(1))
    out["realised_vol_20"] = log_returns.rolling(20, min_periods=20).std()
    return out


def _simulate_trade(
    df: pd.DataFrame,
    signal_pos: int,
    direction: int,
    atr_value: float,
    tp_mult: float,
    sl_mult: float,
    horizon_bars: int,
) -> dict[str, Any] | None:
    entry_pos = signal_pos + 1
    if entry_pos >= len(df):
        return None

    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    entry_price = float(df["open"].iloc[entry_pos])
    tp_dist = float(tp_mult * atr_value)
    sl_dist = float(sl_mult * atr_value)
    scan_stop = min(entry_pos + horizon_bars, len(df))

    for row_pos in range(entry_pos, scan_stop):
        bar_high = float(df["high"].iloc[row_pos])
        bar_low = float(df["low"].iloc[row_pos])

        if direction == 1:
            if bar_high >= entry_price + tp_dist:
                return {"barrier_label": 1, "realized_r": float(tp_mult), "exit_pos": row_pos}
            if bar_low <= entry_price - sl_dist:
                return {"barrier_label": -1, "realized_r": float(-sl_mult), "exit_pos": row_pos}
        else:
            if bar_low <= entry_price - tp_dist:
                return {"barrier_label": 1, "realized_r": float(tp_mult), "exit_pos": row_pos}
            if bar_high >= entry_price + sl_dist:
                return {"barrier_label": -1, "realized_r": float(-sl_mult), "exit_pos": row_pos}

    return {"barrier_label": 0, "realized_r": 0.0, "exit_pos": scan_stop - 1}


def walk_forward_meta_labeling(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_months: int = 6,
    test_months: int = 1,
    purge_gap: int = 10,
    embargo_gap: int = 20,
    risk_per_trade: float = 0.01,
    thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65),
    meta_config: dict[str, Any] | None = None,
    probability_calibration: str = "none",
    sizing_mode: str = "confidence",
) -> Dict:
    """
    Walk-forward validation for the meta-labeling architecture with adaptive
    position sizing and confidence-based stop-loss logic.
    """
    meta_config = dict(meta_config or {})
    clean = _ensure_frame(
        df,
        feature_cols,
        ["base_signal", "reason_code", "barrier_label", "outcome_binary", "realized_r", "atr_20"],
    )
    clean = _prepare_realised_vol(clean)
    clean = clean.dropna(subset=feature_cols).copy()

    signal_rows = clean["base_signal"] != 0
    non_timeout_rows = signal_rows & clean["outcome_binary"].notna()
    max_corr, _ = _compute_feature_target_stats(clean.loc[non_timeout_rows], feature_cols, "outcome_binary")

    tp_mult = float(meta_config.get("tp_atr_mult", 1.5))
    standard_sl_mult = float(meta_config.get("standard_sl_mult", 1.0))
    tight_sl_mult = float(meta_config.get("tight_sl_mult", 0.8))
    adaptive_sl_threshold = float(meta_config.get("adaptive_sl_threshold", 0.60))
    horizon_bars = int(meta_config.get("forward_horizon", 10))
    vol_scale_clamp = tuple(meta_config.get("vol_scale_clamp", (0.25, 2.0)))
    conf_scale_range = tuple(meta_config.get("conf_scale_range", (0.5, 2.0)))
    combined_scale_clamp = tuple(meta_config.get("combined_scale_clamp", (0.2, 3.0)))
    configured_target_vol = meta_config.get("target_vol")

    index_positions = {timestamp: pos for pos, timestamp in enumerate(clean.index)}

    fold_results: list[dict[str, object]] = []
    base_trades: list[dict[str, object]] = []
    filtered_trades: dict[float, list[dict[str, object]]] = {float(threshold): [] for threshold in thresholds}
    is_accuracies: list[float] = []
    oos_accuracies: list[float] = []

    for spec in _iter_fold_boundaries(clean.index, train_months, test_months, purge_gap, embargo_gap):
        fold = int(spec["fold"])
        train_df = clean.iloc[spec["train_slice"]].copy()
        test_df = clean.iloc[spec["test_slice"]].copy()
        train_mask = (train_df["base_signal"] != 0) & train_df["outcome_binary"].notna()
        test_binary_mask = (test_df["base_signal"] != 0) & test_df["outcome_binary"].notna()
        test_signal_mask = (test_df["base_signal"] != 0) & test_df["barrier_label"].notna()

        fold_info: dict[str, object] = {
            "fold": fold,
            "train_start": str(spec["train_start"]),
            "train_end": str(spec["train_end"]),
            "test_start": str(spec["test_start"]),
            "test_end": str(spec["test_end"]),
            "train_signals": int(train_mask.sum()),
            "test_signals": int(test_signal_mask.sum()),
        }

        if int(train_mask.sum()) < 40 or int(test_signal_mask.sum()) < 10:
            fold_info["warning"] = "Insufficient signal rows for fold."
            fold_results.append(fold_info)
            continue

        train_vol_series = pd.to_numeric(train_df.loc[train_mask, "realised_vol_20"], errors="coerce").dropna()
        if configured_target_vol is None:
            target_vol = float(train_vol_series.median()) if not train_vol_series.empty else 1.0
        else:
            target_vol = float(configured_target_vol)
        if not np.isfinite(target_vol) or target_vol <= 0:
            target_vol = 1.0

        full_train_mask = pd.Series(False, index=clean.index)
        full_train_mask.iloc[spec["train_slice"]] = True
        model = train_meta_model(
            clean,
            clean["base_signal"],
            clean["outcome_binary"],
            feature_cols,
            full_train_mask,
        )

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "outcome_binary"].astype(int)
        train_prob = model.predict_proba(X_train)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        is_accuracy = accuracy_score(y_train, train_pred)
        is_accuracies.append(float(is_accuracy))

        # In-sample probability calibration + sizing inputs (no leakage).
        calibrator = _fit_probability_calibrator(
            probability_calibration, train_prob, y_train.to_numpy()
        )
        train_prob_cal = np.asarray(calibrator(train_prob))
        train_r_arr = train_df.loc[train_mask, "realized_r"].astype(float).to_numpy()

        if int(test_binary_mask.sum()) > 0:
            X_test_binary = test_df.loc[test_binary_mask, feature_cols]
            y_test_binary = test_df.loc[test_binary_mask, "outcome_binary"].astype(int)
            test_prob_binary = model.predict_proba(X_test_binary)[:, 1]
            test_pred_binary = (test_prob_binary >= 0.5).astype(int)
            oos_accuracy = accuracy_score(y_test_binary, test_pred_binary)
            oos_accuracies.append(float(oos_accuracy))
        else:
            oos_accuracy = np.nan

        X_test_signals = test_df.loc[test_signal_mask, feature_cols]
        raw_signal_prob = model.predict_proba(X_test_signals)[:, 1]
        signal_prob = pd.Series(
            np.asarray(calibrator(raw_signal_prob)), index=X_test_signals.index, dtype=float
        )

        for timestamp, row in test_df.loc[test_signal_mask].iterrows():
            probability = float(signal_prob.loc[timestamp])
            base_trade = {
                "datetime": timestamp,
                "signal": int(row["base_signal"]),
                "reason_code": row["reason_code"],
                "barrier_label": int(row["barrier_label"]),
                "realized_r": float(row["realized_r"]),
                "pct_return": risk_per_trade * float(row["realized_r"]),
                "pnl_value": risk_per_trade * float(row["realized_r"]),
                "probability": probability,
                "fold": fold,
                "vol_scale": 1.0,
                "conf_scale": 1.0,
                "position_scale": 1.0,
                "target_vol": target_vol,
                "sl_group": "standard",
            }
            base_trades.append(base_trade)

            direction = int(row["base_signal"])
            signal_pos = index_positions[timestamp]
            atr_value = float(row["atr_20"])
            realized_vol = float(row["realised_vol_20"]) if pd.notna(row["realised_vol_20"]) else np.nan
            if np.isfinite(realized_vol) and realized_vol > 0:
                vol_scale = _clamp(target_vol / realized_vol, float(vol_scale_clamp[0]), float(vol_scale_clamp[1]))
            else:
                vol_scale = 1.0

            for threshold in thresholds:
                threshold = float(threshold)
                if probability < threshold:
                    continue

                conf_scale = _position_confidence_multiplier(
                    sizing_mode, probability, threshold, conf_scale_range,
                    train_prob_cal=train_prob_cal, train_r=train_r_arr,
                )
                position_scale = _clamp(
                    vol_scale * conf_scale,
                    float(combined_scale_clamp[0]),
                    float(combined_scale_clamp[1]),
                )
                sl_mult = tight_sl_mult if probability < adaptive_sl_threshold else standard_sl_mult
                sl_group = "tightened" if sl_mult == tight_sl_mult else "standard"
                sim = _simulate_trade(clean, signal_pos, direction, atr_value, tp_mult, sl_mult, horizon_bars)
                if sim is None:
                    continue

                realized_r = float(sim["realized_r"])
                pct_return = risk_per_trade * realized_r * position_scale
                filtered_trades[threshold].append(
                    {
                        "datetime": timestamp,
                        "signal": direction,
                        "reason_code": row["reason_code"],
                        "barrier_label": int(sim["barrier_label"]),
                        "realized_r": realized_r,
                        "pct_return": pct_return,
                        "pnl_value": pct_return,
                        "probability": probability,
                        "fold": fold,
                        "vol_scale": vol_scale,
                        "conf_scale": conf_scale,
                        "position_scale": position_scale,
                        "target_vol": target_vol,
                        "sl_mult_used": sl_mult,
                        "sl_group": sl_group,
                    }
                )

        fold_info.update(
            {
                "is_accuracy": float(is_accuracy),
                "oos_accuracy": float(oos_accuracy) if pd.notna(oos_accuracy) else np.nan,
                "overfitting_ratio": float(is_accuracy / max(float(oos_accuracy), 0.01))
                if pd.notna(oos_accuracy)
                else np.nan,
                "target_vol": target_vol,
            }
        )
        for threshold in thresholds:
            fold_info[f"trades_{float(threshold):.2f}"] = int(
                sum(1 for trade in filtered_trades[float(threshold)] if trade["fold"] == fold)
            )
        fold_results.append(fold_info)

    avg_is_accuracy = float(np.mean(is_accuracies)) if is_accuracies else 0.0
    avg_oos_accuracy = float(np.mean(oos_accuracies)) if oos_accuracies else 0.0
    overfit_ratio = avg_is_accuracy / max(avg_oos_accuracy, 0.01)

    base_metrics = _summarise_trades(base_trades)
    threshold_results: list[dict[str, object]] = []
    best_threshold: float | None = None
    best_metrics: dict[str, object] | None = None
    for threshold in thresholds:
        metrics = _summarise_trades(filtered_trades[float(threshold)])
        result = {"threshold": float(threshold), **metrics}
        threshold_results.append(result)
        if best_metrics is None or _score_threshold_result(metrics) > _score_threshold_result(best_metrics):
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_threshold = None
        best_metrics = _empty_metrics()

    verdict, readiness, checks = _build_top_level_verdict(best_metrics, overfit_ratio)
    passed = int(sum(checks.values()))

    return {
        "verdict": verdict,
        "readiness": readiness,
        "checks": checks,
        "checks_passed": f"{passed}/{len(checks)}",
        "checks_passed_count": passed,
        "base_strategy": base_metrics,
        "threshold_results": threshold_results,
        "best_threshold": best_threshold,
        "best_threshold_result": best_metrics,
        "total_return": best_metrics["total_return"],
        "max_drawdown": best_metrics["max_drawdown"],
        "win_rate": best_metrics["win_rate"],
        "loss_rate": best_metrics["loss_rate"],
        "profit_factor": best_metrics["profit_factor"],
        "sharpe_ratio": best_metrics["sharpe_ratio"],
        "total_trades": best_metrics["total_trades"],
        "avg_is_accuracy": f"{avg_is_accuracy * 100:.1f}%",
        "avg_oos_accuracy": f"{avg_oos_accuracy * 100:.1f}%",
        "overfit_ratio": _format_ratio(overfit_ratio),
        "fold_results": fold_results,
        "filtered_trades": {float(t): filtered_trades[float(t)] for t in thresholds},
        "equity_curve": [],
        "target_type": "meta-labeling on raw rule-based signals",
        "target_definition": TARGET_DEFINITION,
        "purge_gap": purge_gap,
        "embargo_gap": embargo_gap,
        "max_feature_target_corr": max_corr,
        "leakage_risk": _leakage_risk(max_corr),
        "sizing_config": dict(meta_config),
    }


def detect_lookahead_bias(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> List[str]:
    """
    Heuristic leakage checks based on feature names and feature/target correlation.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    warnings_list: list[str] = []
    suspect_words = ("future", "forward", "next", "target", "lead")
    for column in feature_cols:
        lower_name = column.lower()
        for word in suspect_words:
            if word in lower_name:
                warnings_list.append(
                    f"LOOKAHEAD WARNING: feature '{column}' contains suspect word '{word}'."
                )
                break

    clean = df.dropna(subset=[label_col])
    max_corr, correlations = _compute_feature_target_stats(clean, feature_cols, label_col)
    for column, corr in sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True):
        if abs(corr) > 0.30:
            warnings_list.append(
                f"LOOKAHEAD WARNING: feature '{column}' has |corr|={abs(corr):.3f} with '{label_col}'."
            )
    if max_corr > 0.30:
        warnings_list.append(
            "Residual leakage risk is HIGH because at least one feature-target correlation exceeds 0.30."
        )
    if not warnings_list:
        warnings_list.append("No lookahead-bias heuristics triggered.")
    return warnings_list
