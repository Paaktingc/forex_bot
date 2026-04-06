"""
model.py

XGBoost classification model for forex signal prediction.
Provides time-series cross-validation, optional class-weighted training,
model persistence, and live signal prediction.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from collections import Counter
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import config

try:
    from xgboost import XGBClassifier as _PreferredClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    _PreferredClassifier = None
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)
ClassifierType = Any
TRAINING_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 0.3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "objective": "multi:softprob",
    "n_jobs": 4,
}


def _make_classifier(
    n_classes: int,
    *,
    n_estimators: int | None = None,
    use_early_stopping: bool,
) -> ClassifierType:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is required for training but is not available.")

    params = dict(TRAINING_PARAMS)
    params["n_estimators"] = int(n_estimators or TRAINING_PARAMS["n_estimators"])
    params["num_class"] = n_classes
    if use_early_stopping:
        params["early_stopping_rounds"] = 50
    return _PreferredClassifier(**params)


def _coerce_label_encoder(y: np.ndarray, label_encoder: LabelEncoder | None) -> LabelEncoder:
    if label_encoder is not None:
        return label_encoder
    inferred = LabelEncoder()
    inferred.fit(np.unique(y))
    return inferred


def _build_sample_weights(
    y: np.ndarray,
    label_encoder: LabelEncoder,
    *,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None

    counts = Counter(int(label) for label in y)
    n_samples = len(y)
    n_classes = len(counts)
    class_weights = {
        cls: n_samples / (n_classes * count)
        for cls, count in counts.items()
        if count > 0
    }

    for encoded_class, original_label in enumerate(label_encoder.classes_):
        if encoded_class not in class_weights:
            continue
        if int(original_label) == 1:
            class_weights[encoded_class] *= 2.0
        elif int(original_label) == 0:
            class_weights[encoded_class] *= 1.5

    return np.array([class_weights[int(label)] for label in y], dtype=float)


def _run_time_series_cv(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    timestamps: pd.Index | None,
    class_weighted: bool,
    verbose_reports: bool,
) -> dict[str, Any]:
    n_classes = len(np.unique(y))
    target_names = [str(cls) for cls in label_encoder.classes_]
    sample_weights = _build_sample_weights(y, label_encoder, enabled=class_weighted)

    tscv = TimeSeriesSplit(n_splits=5)
    accuracies: list[float] = []
    best_rounds: list[int] = []
    low_sell_recall_folds: list[int] = []
    fold_reports: list[dict[str, Any]] = []

    logger.info("Starting TimeSeriesSplit cross-validation (5 folds) ...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = _make_classifier(n_classes, use_early_stopping=True)
        fit_kwargs: dict[str, Any] = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[train_idx]
            fit_kwargs["sample_weight_eval_set"] = [sample_weights[val_idx]]
        fold_model.fit(X_tr, y_tr, **fit_kwargs)

        y_pred = fold_model.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(float(fold_accuracy))
        report = classification_report(
            y_val,
            y_pred,
            labels=np.arange(len(target_names)),
            target_names=target_names,
            zero_division=0,
        )
        _, recall, _, _ = precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=np.arange(len(target_names)),
            zero_division=0,
        )
        sell_idx = list(label_encoder.classes_).index(-1) if -1 in label_encoder.classes_ else None
        if sell_idx is not None and recall[sell_idx] < 0.40:
            low_sell_recall_folds.append(fold)

        best_iteration = getattr(fold_model, "best_iteration", None)
        best_rounds.append(int(best_iteration) + 1 if best_iteration is not None else TRAINING_PARAMS["n_estimators"])

        if timestamps is not None and len(timestamps) == len(X):
            header = (
                f"Fold {fold}: train {timestamps[train_idx[0]]}→{timestamps[train_idx[-1]]} | "
                f"test {timestamps[val_idx[0]]}→{timestamps[val_idx[-1]]}"
            )
        else:
            header = f"Fold {fold}"

        fold_reports.append(
            {
                "fold": fold,
                "header": header,
                "report": report,
                "accuracy": float(fold_accuracy),
                "y_true": y_val,
                "y_pred": y_pred,
            }
        )
        logger.info("\n===== %s =====\n%s", header, report)
        if verbose_reports:
            print(f"\n{header}")
            print("Classification Report:")
            print(report)
            print(f"Overall accuracy: {fold_accuracy * 100:.2f}%")

    average_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
    if verbose_reports:
        print(f"\nAverage accuracy across folds: {average_accuracy * 100:.2f}%")
        if low_sell_recall_folds:
            print(
                "WARNING: sell class recall < 0.40 on fold(s): "
                + ", ".join(str(fold) for fold in low_sell_recall_folds)
            )

    if average_accuracy > 0.80:
        raise RuntimeError(
            "⛔ STOPPED: Average fold accuracy >80% — likely lookahead bias or data leakage.\n"
            "Check: features built without future data, no label leakage, no train/test overlap."
        )

    return {
        "average_accuracy": average_accuracy,
        "best_rounds": best_rounds,
        "fold_reports": fold_reports,
        "low_sell_recall_folds": low_sell_recall_folds,
    }


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    timestamps: pd.Index | None = None,
    *,
    class_weighted: bool = False,
    persist: bool = True,
    verbose_reports: bool = True,
) -> ClassifierType:
    """
    Trains an XGBClassifier using walk-forward TimeSeriesSplit (never shuffled).
    """
    label_encoder = _coerce_label_encoder(y, label_encoder)
    cv_result = _run_time_series_cv(
        X=X,
        y=y,
        label_encoder=label_encoder,
        timestamps=timestamps,
        class_weighted=class_weighted,
        verbose_reports=verbose_reports,
    )

    final_rounds = int(round(np.mean(cv_result["best_rounds"]))) if cv_result["best_rounds"] else TRAINING_PARAMS["n_estimators"]
    final_rounds = max(10, final_rounds)
    logger.info("Training final model on full dataset with %s estimators ...", final_rounds)

    final_model = _make_classifier(
        len(np.unique(y)),
        n_estimators=final_rounds,
        use_early_stopping=False,
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    full_weights = _build_sample_weights(y, label_encoder, enabled=class_weighted)
    if full_weights is not None:
        fit_kwargs["sample_weight"] = full_weights
    final_model.fit(X, y, **fit_kwargs)

    if persist:
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        with open(config.MODEL_PATH, "wb") as f:
            pickle.dump(final_model, f)
        logger.info("Model saved to %s", config.MODEL_PATH)

    setattr(final_model, "_cv_result", cv_result)
    setattr(final_model, "_feature_names", list(X.columns))
    return final_model


def evaluate_model(
    model: ClassifierType,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """
    Evaluates the model on a held-out test set.
    """
    y_pred = model.predict(X_test)

    target_names = [str(c) for c in le.classes_]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n=== Evaluation Report ===\n%s", report)
    logger.info("Confusion Matrix:\n%s", cm)
    print("\n=== Evaluation Report ===")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )

    results: dict[str, float] = {"accuracy": acc}
    for i, cls in enumerate(le.classes_):
        results[f"precision_{cls}"] = precision[i]
        results[f"recall_{cls}"] = recall[i]
        results[f"f1_{cls}"] = f1[i]

    if -1 in le.classes_:
        sell_idx = list(le.classes_).index(-1)
        if recall[sell_idx] < 0.40:
            logger.warning(
                "Sell class recall is %.2f — below 0.40 threshold. Model may under-predict sell signals.",
                recall[sell_idx],
            )

    return results


def predict_signal(
    model: ClassifierType,
    le: LabelEncoder,
    X_live: pd.DataFrame,
) -> Tuple[int, float]:
    """
    Predicts a trading signal from live feature data.
    """
    try:
        proba = model.predict_proba(X_live)[0]
        confidence = float(np.max(proba))
        predicted_encoded = int(np.argmax(proba))
        signal = int(le.inverse_transform([predicted_encoded])[0])

        if confidence < config.MIN_CONFIDENCE:
            return (0, confidence)

        return (signal, confidence)
    except Exception as exc:
        logger.error("Error during predict_signal: %s", exc)
        return (0, 0.0)


def load_model() -> Tuple[ClassifierType, LabelEncoder]:
    """
    Loads model and label encoder from disk.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    if not os.path.exists(config.ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {config.ENCODER_PATH}")

    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(config.ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba.")

    logger.info("Model and LabelEncoder loaded successfully.")
    return model, le


def _load_training_inputs() -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, pd.Index]:
    X = pd.read_csv(
        config.DATA_DIR / "X_train.csv",
        index_col=0,
        parse_dates=True,
    )
    y = np.load(config.DATA_DIR / "y_train.npy")
    with open(config.ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    if isinstance(X.index, pd.DatetimeIndex) and len(X.index) == len(X):
        if X.index.tz is None:
            timestamps = X.index.tz_localize("UTC")
        else:
            timestamps = X.index.tz_convert("UTC")
    else:
        feature_timeline = pd.read_csv(
            config.DATA_DIR / "EURUSD_features.csv",
            index_col=0,
            parse_dates=True,
        )
        if feature_timeline.index.tz is None:
            feature_timeline.index = feature_timeline.index.tz_localize("UTC")
        else:
            feature_timeline.index = feature_timeline.index.tz_convert("UTC")
        if len(feature_timeline) < len(X):
            raise ValueError(
                "EURUSD_features.csv has fewer rows than X_train.csv; cannot derive fold date ranges."
            )
        timestamps = feature_timeline.index[: len(X)]

    return X, y, le, timestamps


def _save_feature_importance_plot(model: ClassifierType, feature_names: list[str]) -> pd.Series | None:
    if not hasattr(model, "feature_importances_"):
        print("\nFeature importance not available for this model.")
        return None

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTOP 15 FEATURE IMPORTANCE")
    print("=" * 50)
    for feature, value in importance.head(15).items():
        print(f"{feature:<18} {float(value):.6f}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        importance.head(20).sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_title("Top 20 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        output_path = config.LOGS_DIR / "feature_importance.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Feature importance plot saved: {output_path}")
    except Exception as exc:
        logger.warning("Failed to save feature importance plot: %s", exc)

    return importance


def _print_session_feature_ranking(importance: pd.Series | None) -> None:
    if importance is None or importance.empty:
        return

    session_features = ["is_london", "is_ny", "is_overlap"]
    price_features = [
        "atr_14",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "price_momentum",
        "volatility_ratio",
        "bb_position",
        "ema_20",
        "ema_50",
        "adx_14",
    ]
    available_prices = [f for f in price_features if f in importance.index]
    if not available_prices:
        return

    strongest_price = available_prices[0]
    strongest_rank = importance.index.get_loc(strongest_price) + 1
    session_ranks = {
        feature: importance.index.get_loc(feature) + 1
        for feature in session_features
        if feature in importance.index
    }
    all_below = all(rank > strongest_rank for rank in session_ranks.values())
    print(
        "Session feature rank check: "
        f"strongest price/volatility feature is {strongest_price} (rank {strongest_rank}); "
        f"session ranks = {session_ranks or 'n/a'}"
    )
    print("Session features below strongest price feature: " + ("YES" if all_below else "NO"))


def _print_last_fold_targets(model: ClassifierType, label_encoder: LabelEncoder) -> None:
    cv_result = getattr(model, "_cv_result", None)
    if not cv_result or not cv_result.get("fold_reports"):
        return

    last_fold = cv_result["fold_reports"][-1]
    print(f"\n{last_fold['header']}")
    print("Classification Report:")
    print(last_fold["report"])

    _, recall, _, _ = precision_recall_fscore_support(
        last_fold["y_true"],
        last_fold["y_pred"],
        labels=np.arange(len(label_encoder.classes_)),
        zero_division=0,
    )
    recall_by_label = {
        int(label_encoder.classes_[idx]): float(recall[idx])
        for idx in range(len(label_encoder.classes_))
    }
    print("Last fold recall targets")
    print("=" * 50)
    print(f"SELL recall (-1): {recall_by_label.get(-1, 0.0):.2f} | target 0.75-0.85")
    print(f"HOLD recall (0) : {recall_by_label.get(0, 0.0):.2f} | target 0.50+")
    print(f"BUY recall (1)  : {recall_by_label.get(1, 0.0):.2f} | target 0.70+")


def run_training_pipeline(*, class_weighted: bool = False) -> None:
    """
    Trains XGBoost on prepared X/y training arrays using TimeSeriesSplit CV.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is not importable in this environment.")

    start_time = time.perf_counter()
    X, y, le, timestamps = _load_training_inputs()
    logger.info("Loaded training data — X shape: %s | y shape: %s", X.shape, y.shape)

    model = train_model(
        X,
        y,
        label_encoder=le,
        timestamps=timestamps,
        class_weighted=class_weighted,
        persist=True,
        verbose_reports=True,
    )

    importance = _save_feature_importance_plot(model, list(X.columns))
    _print_session_feature_ranking(importance)
    if class_weighted:
        _print_last_fold_targets(model, le)

    elapsed = time.perf_counter() - start_time
    print("\nFINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Mode          : {'class-weighted' if class_weighted else 'regularized'}")
    print(f"Training time : {elapsed:.2f}s")
    print(f"Model saved   : {config.MODEL_PATH}")
    print(f"Encoder saved : {config.ENCODER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EURUSD XGBoost model.")
    parser.add_argument(
        "--class-weighted",
        action="store_true",
        help="Train with per-sample class weights for hold/buy emphasis.",
    )
    args = parser.parse_args()
    run_training_pipeline(class_weighted=args.class_weighted)
