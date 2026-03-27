"""
model.py

XGBoost classification model for forex signal prediction.
Provides training with TimeSeriesSplit cross-validation,
evaluation with per-class metrics, live signal prediction,
model persistence, and a complete training pipeline.
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
import sys
from typing import Any, Tuple

try:
    from xgboost import XGBClassifier as _PreferredClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    _PreferredClassifier = None
    XGBOOST_AVAILABLE = False
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

import config

logger = logging.getLogger(__name__)

ClassifierType = Any

if sys.platform == "darwin":
    XGBOOST_AVAILABLE = False


def _make_classifier(n_classes: int) -> ClassifierType:
    if XGBOOST_AVAILABLE:
        return _PreferredClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            objective="multi:softprob",
            num_class=n_classes,
        )

    logger.warning("XGBoost unavailable. Falling back to DecisionTreeClassifier.")
    return DecisionTreeClassifier(
        max_depth=6,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────
# 1. train_model
# ─────────────────────────────────────────────────────────────
def train_model(X: pd.DataFrame, y: np.ndarray) -> ClassifierType:
    """
    Trains an XGBClassifier using walk-forward TimeSeriesSplit (never shuffled).

    Steps:
        1. 5-fold time-series cross-validation with early stopping.
        2. Prints classification_report per fold.
        3. Retrains final model on the full dataset.
        4. Saves model to config.MODEL_PATH.

    Args:
        X: Feature DataFrame (rows aligned with y).
        y: Encoded target array.

    Returns:
        XGBClassifier: The final trained model.
    """
    n_classes = len(np.unique(y))

    tscv = TimeSeriesSplit(n_splits=5)

    logger.info("Starting TimeSeriesSplit cross-validation (5 folds) ...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = _make_classifier(n_classes)
        fit_kwargs = {}
        if XGBOOST_AVAILABLE:
            fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        fold_model.fit(X_tr, y_tr, **fit_kwargs)

        y_pred = fold_model.predict(X_val)
        report = classification_report(y_val, y_pred, zero_division=0)
        logger.info(f"\n===== Fold {fold} =====\n{report}")
        print(f"\n===== Fold {fold} =====")
        print(report)

    # Final model on full dataset (no eval_set / early stopping)
    logger.info("Training final model on full dataset ...")
    final_model = _make_classifier(n_classes)
    final_fit_kwargs = {"verbose": False} if XGBOOST_AVAILABLE else {}
    final_model.fit(X, y, **final_fit_kwargs)

    # Persist
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"Model saved to {config.MODEL_PATH}")

    return final_model


# ─────────────────────────────────────────────────────────────
# 2. evaluate_model
# ─────────────────────────────────────────────────────────────
def evaluate_model(
    model: ClassifierType,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """
    Evaluates the model on a held-out test set.

    Prints a classification report and confusion matrix.
    Logs a WARNING if the sell-class recall falls below 0.40.

    Args:
        model: Trained XGBClassifier.
        X_test: Test features.
        y_test: Encoded test labels.
        le: LabelEncoder used during training.

    Returns:
        dict: Per-class accuracy, precision, recall, f1.
    """
    y_pred = model.predict(X_test)

    target_names = [str(c) for c in le.classes_]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n=== Evaluation Report ===\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")
    print("\n=== Evaluation Report ===")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )

    results: dict = {"accuracy": acc}
    for i, cls in enumerate(le.classes_):
        results[f"precision_{cls}"] = precision[i]
        results[f"recall_{cls}"] = recall[i]
        results[f"f1_{cls}"] = f1[i]

    # Warn if sell-class recall is weak
    sell_label = -1
    if sell_label in le.classes_:
        sell_idx = list(le.classes_).index(sell_label)
        if recall[sell_idx] < 0.40:
            logger.warning(
                f"Sell class recall is {recall[sell_idx]:.2f} — below 0.40 threshold. "
                "Model may under-predict sell signals."
            )

    return results


# ─────────────────────────────────────────────────────────────
# 3. predict_signal
# ─────────────────────────────────────────────────────────────
def predict_signal(
    model: ClassifierType,
    le: LabelEncoder,
    X_live: pd.DataFrame,
) -> Tuple[int, float]:
    """
    Predicts a trading signal from live feature data.

    Args:
        model: Trained XGBClassifier.
        le: LabelEncoder mapping encoded → original labels.
        X_live: Single-row (or last-row) feature DataFrame.

    Returns:
        (signal, confidence):
            signal — original label (1 = buy, -1 = sell, 0 = hold)
            confidence — maximum class probability
            Returns (0, confidence) when confidence < MIN_CONFIDENCE.
    """
    try:
        proba = model.predict_proba(X_live)[0]
        confidence = float(np.max(proba))
        predicted_encoded = int(np.argmax(proba))
        signal = int(le.inverse_transform([predicted_encoded])[0])

        if confidence < config.MIN_CONFIDENCE:
            return (0, confidence)

        return (signal, confidence)

    except Exception as e:
        logger.error(f"Error during predict_signal: {e}")
        return (0, 0.0)


# ─────────────────────────────────────────────────────────────
# 4. load_model
# ─────────────────────────────────────────────────────────────
def load_model() -> Tuple[ClassifierType, LabelEncoder]:
    """
    Loads model and label encoder from disk.

    Returns:
        (model, le): Trained XGBClassifier and fitted LabelEncoder.

    Raises:
        FileNotFoundError: If model or encoder files are missing.
        AttributeError: If loaded model lacks predict_proba.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    if not os.path.exists(config.ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {config.ENCODER_PATH}")

    if sys.platform == "darwin":
        with open(config.MODEL_PATH, "rb") as f:
            header = f.read(4096)
        if b"xgboost" in header.lower():
            raise RuntimeError(
                "Saved XGBoost model is not supported in this Mac/dev environment."
            )

    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(config.ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba.")
    if sys.platform == "darwin" and model.__class__.__module__.startswith("xgboost"):
        raise RuntimeError(
            "Saved XGBoost model is not supported in this Mac/dev environment."
        )

    logger.info("Model and LabelEncoder loaded successfully.")
    return model, le


# ─────────────────────────────────────────────────────────────
# 5. run_training_pipeline
# ─────────────────────────────────────────────────────────────
def run_training_pipeline() -> None:
    """
    End-to-end training pipeline:
        1. Loads data/X_train.csv and data/y_train.npy.
        2. Trains via TimeSeriesSplit CV.
        3. Evaluates on the last 20% of the data.
        4. Prints a final performance summary.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    X_path = config.DATA_DIR / "X_train.csv"
    y_path = config.DATA_DIR / "y_train.npy"

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logger.error(
            f"Training data not found. Expected:\n  {X_path}\n  {y_path}\n"
            "Run labelling.py first to generate prepared data."
        )
        return

    logger.info("Loading training data ...")
    X = pd.read_csv(X_path)
    y = np.load(y_path)

    if not os.path.exists(config.ENCODER_PATH):
        logger.error(f"LabelEncoder not found at {config.ENCODER_PATH}")
        return

    with open(config.ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    logger.info(f"Data loaded — X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Classes: {le.classes_}")

    # Train
    model = train_model(X, y)

    # Evaluate on last 20%
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]

    results = evaluate_model(model, X_test, y_test, le)

    print("\n" + "=" * 50)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Accuracy : {results['accuracy']:.4f}")
    for cls in le.classes_:
        print(f"  Class {cls:>3d}:")
        print(f"    Precision : {results[f'precision_{cls}']:.4f}")
        print(f"    Recall    : {results[f'recall_{cls}']:.4f}")
        print(f"    F1        : {results[f'f1_{cls}']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_training_pipeline()
