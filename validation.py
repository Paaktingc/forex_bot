"""
validation.py

Causally clean walk-forward validation with purge/embargo separation.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from meta_labeling import apply_meta_filter, train_meta_model

logger = logging.getLogger(__name__)


def _ensure_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    extra_columns: Iterable[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Validation requires a DatetimeIndex.")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column.")

    required = ["open", "high", "low", "close", "volume", *feature_cols, *extra_columns]
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


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray | None:
    if len(y) == 0:
        return None
    labels, counts = np.unique(y, return_counts=True)
    max_share = float(counts.max() / counts.sum())
    if max_share <= 0.60:
        return None

    weights = {label: len(y) / (len(labels) * count) for label, count in zip(labels, counts, strict=False)}
    return np.array([weights[label] for label in y], dtype=float)


def _build_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=30,
        subsample=0.8,
        random_state=42,
    )


def _iter_fold_boundaries(
    index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    purge_bars: int,
    embargo_bars: int,
) -> list[dict[str, object]]:
    if len(index) == 0:
        return []

    fold_specs: list[dict[str, object]] = []
    start_date = index.min()
    end_date = index.max()
    train_end_date = start_date + pd.DateOffset(months=train_months)
    fold = 0

    while train_end_date < end_date:
        fold += 1
        train_end_pos = int(index.searchsorted(train_end_date, side="left"))
        test_start_pos = train_end_pos + purge_bars + embargo_bars
        test_end_date = train_end_date + pd.DateOffset(months=test_months)
        test_end_pos = int(index.searchsorted(test_end_date, side="left"))

        if train_end_pos <= 0:
            train_end_date += pd.DateOffset(months=test_months)
            continue
        if test_start_pos >= len(index):
            break
        if test_end_pos <= test_start_pos:
            train_end_date += pd.DateOffset(months=test_months)
            continue

        fold_specs.append(
            {
                "fold": fold,
                "train_slice": slice(0, train_end_pos),
                "test_slice": slice(test_start_pos, test_end_pos),
                "train_start": index[0],
                "train_end": index[train_end_pos - 1],
                "test_start": index[test_start_pos],
                "test_end": index[test_end_pos - 1],
            }
        )
        train_end_date += pd.DateOffset(months=test_months)

    return fold_specs


def _simulate_trade_from_targets(
    row: pd.Series,
    signal: int,
    risk_per_trade: float,
) -> dict[str, object] | None:
    if signal == 0:
        return None

    target_column = "target_long" if signal == 1 else "target_short"
    if target_column not in row or pd.isna(row[target_column]):
        return None

    win = bool(int(row[target_column]) == 1)
    pnl = risk_per_trade if win else -risk_per_trade
    return {"win": win, "pnl": pnl, "prediction": int(signal)}


def _aggregate_results(
    *,
    all_trades: list[dict[str, object]],
    fold_results: list[dict[str, object]],
    purge_bars: int,
    embargo_bars: int,
    max_corr: float,
    target_type: str,
    return_target: float,
    max_dd_target: float,
    profit_factor_target: float,
    sharpe_target: float,
    overfit_target: float,
) -> dict[str, object]:
    if not all_trades:
        checks = {
            "return_positive": False,
            "return_target": False,
            "max_dd_ok": False,
            "win_rate_ok": False,
            "profit_factor_ok": False,
            "sharpe_ok": False,
            "enough_trades": False,
            "not_overfit": False,
        }
        return {
            "verdict": "NO-GO",
            "reason": "No OOS trades generated",
            "checks": checks,
            "checks_passed": "0/8",
            "checks_passed_count": 0,
            "total_return": "0.00%",
            "max_drawdown": "0.00%",
            "win_rate": "0.0%",
            "profit_factor": "0.00",
            "sharpe_ratio": "0.00",
            "total_trades": 0,
            "avg_is_accuracy": "0.0%",
            "avg_oos_accuracy": "0.0%",
            "overfit_ratio": "0.00",
            "fold_results": fold_results,
            "equity_curve": [],
            "target_type": target_type,
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "max_feature_target_corr": max_corr,
            "leakage_risk": _leakage_risk(max_corr),
        }

    trades_df = pd.DataFrame(all_trades).sort_values("datetime")
    cumulative_pnl = trades_df["pnl"].cumsum()
    drawdown = cumulative_pnl - cumulative_pnl.cummax()

    total_return = float(cumulative_pnl.iloc[-1])
    max_drawdown = float(drawdown.min())
    total_trades = int(len(trades_df))
    total_wins = int(trades_df["win"].sum())
    win_rate = total_wins / total_trades if total_trades else 0.0

    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss = abs(float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_returns = trades_df.set_index("datetime")["pnl"].resample("1D").sum()
    daily_returns = daily_returns[daily_returns != 0]
    if len(daily_returns) > 1 and float(daily_returns.std()) > 0:
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    is_accuracies = [float(fold["is_accuracy"]) for fold in fold_results if "is_accuracy" in fold]
    oos_accuracies = [float(fold["oos_accuracy"]) for fold in fold_results if "oos_accuracy" in fold]
    avg_is_accuracy = float(np.mean(is_accuracies)) if is_accuracies else 0.0
    avg_oos_accuracy = float(np.mean(oos_accuracies)) if oos_accuracies else 0.0
    overfit_ratio = avg_is_accuracy / max(avg_oos_accuracy, 0.01)

    checks = {
        "return_positive": total_return > 0.0,
        "return_target": total_return > return_target,
        "max_dd_ok": abs(max_drawdown) < max_dd_target,
        "win_rate_ok": win_rate > 0.45,
        "profit_factor_ok": profit_factor > profit_factor_target,
        "sharpe_ok": sharpe_ratio > sharpe_target,
        "enough_trades": total_trades > 50,
        "not_overfit": overfit_ratio < overfit_target,
    }
    checks_passed_count = int(sum(checks.values()))
    if checks_passed_count >= 7:
        verdict = "GO"
    elif checks_passed_count >= 5:
        verdict = "CONDITIONAL"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "checks": checks,
        "checks_passed": f"{checks_passed_count}/8",
        "checks_passed_count": checks_passed_count,
        "total_return": f"{total_return * 100:.2f}%",
        "max_drawdown": f"{max_drawdown * 100:.2f}%",
        "win_rate": f"{win_rate * 100:.1f}%",
        "profit_factor": f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "inf",
        "sharpe_ratio": f"{sharpe_ratio:.2f}",
        "total_trades": total_trades,
        "avg_is_accuracy": f"{avg_is_accuracy * 100:.1f}%",
        "avg_oos_accuracy": f"{avg_oos_accuracy * 100:.1f}%",
        "overfit_ratio": f"{overfit_ratio:.2f}",
        "fold_results": fold_results,
        "equity_curve": cumulative_pnl.tolist(),
        "target_type": target_type,
        "purge_bars": purge_bars,
        "embargo_bars": embargo_bars,
        "max_feature_target_corr": max_corr,
        "leakage_risk": _leakage_risk(max_corr),
    }


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "target_best",
    train_months: int = 6,
    test_months: int = 1,
    purge_bars: int = 24,
    embargo_bars: int = 12,
    risk_per_trade: float = 0.01,
    atr_col: str = "atr_14",
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
) -> Dict:
    """
    Expanding-window walk-forward validation with purge and embargo gaps.

    Training uses only bars where ``label_col != 0``. Actual PnL for a predicted
    direction is sourced from the causally clean ``target_long`` / ``target_short``
    columns, which were built from next-bar entry and future-only trade paths.
    """
    clean = _ensure_frame(df, feature_cols, [label_col, atr_col, "target_long", "target_short"])
    clean = clean.dropna(subset=[*feature_cols, label_col, atr_col, "target_long", "target_short"]).copy()
    max_corr, _ = _compute_feature_target_stats(clean, feature_cols, label_col)

    fold_results: list[dict[str, object]] = []
    all_trades: list[dict[str, object]] = []
    fold_specs = _iter_fold_boundaries(clean.index, train_months, test_months, purge_bars, embargo_bars)

    for spec in fold_specs:
        fold = int(spec["fold"])
        train_df = clean.iloc[spec["train_slice"]].copy()
        test_df = clean.iloc[spec["test_slice"]].copy()
        fold_info: dict[str, object] = {
            "fold": fold,
            "train_start": str(spec["train_start"]),
            "train_end": str(spec["train_end"]),
            "test_start": str(spec["test_start"]),
            "test_end": str(spec["test_end"]),
        }

        train_mask = train_df[label_col].astype(int) != 0
        test_label_mask = test_df[label_col].astype(int) != 0
        if int(train_mask.sum()) < 40 or len(test_df) < 10:
            fold_info["warning"] = "Insufficient training/test rows after purge and embargo."
            fold_results.append(fold_info)
            continue
        if int(test_label_mask.sum()) < 10:
            fold_info["warning"] = "Insufficient directional labels in test fold."
            fold_results.append(fold_info)
            continue

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, label_col].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            fold_info["warning"] = "Training fold contains fewer than two directional classes."
            fold_results.append(fold_info)
            continue

        model = _build_model()
        sample_weight = _balanced_sample_weights(y_train)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)

        is_pred = model.predict(X_train)
        is_accuracy = accuracy_score(y_train, is_pred)

        X_test = test_df[feature_cols]
        predicted_signal = pd.Series(model.predict(X_test), index=test_df.index, dtype=int)
        oos_accuracy = accuracy_score(
            test_df.loc[test_label_mask, label_col].astype(int),
            predicted_signal.loc[test_label_mask].astype(int),
        )

        fold_trades: list[dict[str, object]] = []
        for timestamp, row in test_df.iterrows():
            signal = int(predicted_signal.loc[timestamp])
            trade = _simulate_trade_from_targets(row, signal, risk_per_trade)
            if trade is None:
                continue
            trade["datetime"] = timestamp
            trade["actual"] = signal
            trade["fold"] = fold
            fold_trades.append(trade)

        all_trades.extend(fold_trades)
        fold_info.update(
            {
                "is_accuracy": float(is_accuracy),
                "oos_accuracy": float(oos_accuracy),
                "oos_trades": int(len(fold_trades)),
                "oos_wins": int(sum(1 for trade in fold_trades if trade["win"])),
                "overfitting_ratio": float(is_accuracy / max(oos_accuracy, 0.01)),
                "tp_atr_mult": tp_atr_mult,
                "sl_atr_mult": sl_atr_mult,
            }
        )
        fold_results.append(fold_info)

    return _aggregate_results(
        all_trades=all_trades,
        fold_results=fold_results,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        max_corr=max_corr,
        target_type="trade-return",
        return_target=0.03,
        max_dd_target=0.10,
        profit_factor_target=1.2,
        sharpe_target=0.5,
        overfit_target=1.8,
    )


def walk_forward_meta_labeling(
    df: pd.DataFrame,
    feature_cols: List[str],
    base_signals: pd.Series,
    signal_outcomes: pd.Series,
    train_months: int = 6,
    test_months: int = 1,
    purge_bars: int = 24,
    embargo_bars: int = 12,
    risk_per_trade: float = 0.01,
    confidence_threshold: float = 0.55,
) -> Dict:
    """
    Walk-forward evaluation for the meta-labeling path.
    """
    clean = _ensure_frame(df, feature_cols, ["target_long", "target_short"])
    clean["base_signal"] = base_signals.reindex(clean.index)
    clean["signal_outcome"] = signal_outcomes.reindex(clean.index)
    clean = clean.dropna(subset=[*feature_cols, "base_signal"]).copy()
    max_corr, _ = _compute_feature_target_stats(
        clean.dropna(subset=["signal_outcome"]),
        feature_cols,
        "signal_outcome",
    )

    fold_results: list[dict[str, object]] = []
    all_trades: list[dict[str, object]] = []
    fold_specs = _iter_fold_boundaries(clean.index, train_months, test_months, purge_bars, embargo_bars)

    for spec in fold_specs:
        fold = int(spec["fold"])
        train_df = clean.iloc[spec["train_slice"]].copy()
        test_df = clean.iloc[spec["test_slice"]].copy()
        train_mask = pd.Series(True, index=clean.index)
        train_mask.loc[:] = False
        train_mask.iloc[spec["train_slice"]] = True

        eligible_train = (
            (train_df["base_signal"].astype(int) != 0)
            & train_df["signal_outcome"].notna()
        )
        eligible_test = (
            (test_df["base_signal"].astype(int) != 0)
            & test_df["signal_outcome"].notna()
        )
        fold_info: dict[str, object] = {
            "fold": fold,
            "train_start": str(spec["train_start"]),
            "train_end": str(spec["train_end"]),
            "test_start": str(spec["test_start"]),
            "test_end": str(spec["test_end"]),
        }

        if int(eligible_train.sum()) < 40 or int(eligible_test.sum()) < 10:
            fold_info["warning"] = "Insufficient signalled bars for meta-labeling fold."
            fold_results.append(fold_info)
            continue

        model = train_meta_model(
            clean,
            clean["base_signal"],
            clean["signal_outcome"],
            feature_cols,
            train_mask,
        )

        train_probs = model.predict_proba(train_df.loc[eligible_train, feature_cols])[:, 1]
        train_pred = (train_probs >= confidence_threshold).astype(int)
        is_accuracy = accuracy_score(
            train_df.loc[eligible_train, "signal_outcome"].astype(int),
            train_pred,
        )

        test_probs = model.predict_proba(test_df.loc[eligible_test, feature_cols])[:, 1]
        oos_pred = (test_probs >= confidence_threshold).astype(int)
        oos_accuracy = accuracy_score(
            test_df.loc[eligible_test, "signal_outcome"].astype(int),
            oos_pred,
        )

        filtered_signals = apply_meta_filter(
            model,
            test_df,
            test_df["base_signal"],
            feature_cols,
            confidence_threshold=confidence_threshold,
        )

        fold_trades: list[dict[str, object]] = []
        for timestamp, row in test_df.iterrows():
            signal = int(filtered_signals.loc[timestamp])
            if signal == 0:
                continue
            outcome = row["signal_outcome"]
            if pd.isna(outcome):
                continue
            win = bool(int(outcome) == 1)
            pnl = risk_per_trade if win else -risk_per_trade
            fold_trades.append(
                {
                    "datetime": timestamp,
                    "prediction": signal,
                    "actual": int(row["base_signal"]),
                    "pnl": pnl,
                    "win": win,
                    "fold": fold,
                }
            )

        all_trades.extend(fold_trades)
        fold_info.update(
            {
                "is_accuracy": float(is_accuracy),
                "oos_accuracy": float(oos_accuracy),
                "oos_trades": int(len(fold_trades)),
                "oos_wins": int(sum(1 for trade in fold_trades if trade["win"])),
                "overfitting_ratio": float(is_accuracy / max(oos_accuracy, 0.01)),
                "confidence_threshold": confidence_threshold,
            }
        )
        fold_results.append(fold_info)

    return _aggregate_results(
        all_trades=all_trades,
        fold_results=fold_results,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        max_corr=max_corr,
        target_type="trade-return meta-label",
        return_target=0.03,
        max_dd_target=0.10,
        profit_factor_target=1.2,
        sharpe_target=0.5,
        overfit_target=1.8,
    )


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

    max_corr, correlations = _compute_feature_target_stats(df, feature_cols, label_col)
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
