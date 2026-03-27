"""
test_model.py

Unit tests for the model.py module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import Stage1Model
import config

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    rows = 200
    df = pd.DataFrame({
        'feature1': np.random.randn(rows),
        'feature2': np.random.randn(rows),
        'target': np.random.choice([0, 1, 2], size=rows)
    })
    return df

def test_stage1_model_training(dummy_data):
    model = Stage1Model()
    model.train(dummy_data)
    assert model.is_trained is True
    assert 'feature1' in model.features
    assert 'feature2' in model.features

def test_stage1_model_prediction(dummy_data, monkeypatch):
    # Lower confidence threshold for test predictability
    monkeypatch.setattr(config, 'CONFIDENCE_THRESHOLD', 0.1)
    
    model = Stage1Model()
    model.train(dummy_data)
    
    test_df = pd.DataFrame({
        'feature1': [0.5],
        'feature2': [-0.5],
        'target': [1] # Should be ignored during predict
    })
    
    pred = model.predict(test_df)
    assert 'signal' in pred
    assert 'confidence' in pred
    assert pred['signal'] in [0, 1, 2]
    assert 0.0 <= pred['confidence'] <= 1.0

def test_model_save_load(dummy_data, tmp_path):
    model = Stage1Model()
    model.train(dummy_data)
    
    filepath = tmp_path / "test_xgb.pkl"
    model.save(str(filepath))
    
    assert os.path.exists(filepath)
    
    new_model = Stage1Model()
    new_model.load(str(filepath))
    
    assert new_model.is_trained is True
    assert new_model.features == model.features
