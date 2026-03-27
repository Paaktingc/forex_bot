"""
test_model.py

Unit tests for model.py functions.
"""

import pytest
import numpy as np
import pandas as pd
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from model import train_model, evaluate_model, predict_signal, load_model


# ── fixtures ──
@pytest.fixture
def dummy_Xy():
    """Creates a small reproducible dataset for testing."""
    np.random.seed(42)
    rows = 200
    X = pd.DataFrame({
        'f1': np.random.randn(rows),
        'f2': np.random.randn(rows),
        'f3': np.random.randn(rows),
    })
    y = np.random.choice([0, 1, 2], size=rows)
    return X, y


@pytest.fixture
def trained_artifacts(dummy_Xy, tmp_path):
    """Trains a model and saves it + a fake encoder to tmp_path for loading tests."""
    X, y = dummy_Xy
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit([-1, 0, 1])

    # Temporarily override config paths
    model_path = str(tmp_path / "model.pkl")
    encoder_path = str(tmp_path / "label_encoder.pkl")

    with patch.object(config, "MODEL_PATH", model_path):
        model = train_model(X, y)

    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    return model, le, model_path, encoder_path


# ── train_model ──
def test_train_model_returns_fitted_model(dummy_Xy, tmp_path):
    X, y = dummy_Xy
    with patch.object(config, "MODEL_PATH", str(tmp_path / "model.pkl")):
        model = train_model(X, y)

    assert hasattr(model, "predict_proba")
    assert os.path.exists(tmp_path / "model.pkl")


# ── evaluate_model ──
def test_evaluate_model_returns_metrics(trained_artifacts):
    from sklearn.preprocessing import LabelEncoder
    model, le, _, _ = trained_artifacts

    np.random.seed(0)
    X_test = pd.DataFrame({'f1': np.random.randn(50),
                           'f2': np.random.randn(50),
                           'f3': np.random.randn(50)})
    y_test = np.random.choice([0, 1, 2], size=50)

    results = evaluate_model(model, X_test, y_test, le)

    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0


# ── predict_signal ──
def test_predict_signal_returns_tuple(trained_artifacts):
    model, le, _, _ = trained_artifacts

    X_live = pd.DataFrame({'f1': [0.5], 'f2': [-0.3], 'f3': [1.0]})
    signal, confidence = predict_signal(model, le, X_live)

    assert isinstance(signal, int)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_predict_signal_low_confidence(trained_artifacts, monkeypatch):
    """When MIN_CONFIDENCE is very high, signal should be 0."""
    model, le, _, _ = trained_artifacts
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.99)

    X_live = pd.DataFrame({'f1': [0.5], 'f2': [-0.3], 'f3': [1.0]})
    signal, confidence = predict_signal(model, le, X_live)

    # With random data, confidence is unlikely ≥ 0.99
    assert signal == 0


# ── load_model ──
def test_load_model_success(trained_artifacts):
    _, _, model_path, encoder_path = trained_artifacts

    with patch.object(config, "MODEL_PATH", model_path), \
         patch.object(config, "ENCODER_PATH", encoder_path):
        m, enc = load_model()

    assert hasattr(m, "predict_proba")
    assert hasattr(enc, "inverse_transform")


def test_load_model_missing_file():
    with patch.object(config, "MODEL_PATH", "/nonexistent/model.pkl"):
        with pytest.raises(FileNotFoundError):
            load_model()
