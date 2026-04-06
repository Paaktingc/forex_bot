import pytest
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import config
from labelling import (
    apply_triple_barrier,
    get_label_distribution,
    prepare_training_data,
    save_prepared_data
)

@pytest.fixture
def synthetic_data():
    """
    Creates a synthetic DataFrame for testing.
    Prices start at 100.
    ATR is constant at 1.0.
    Upper bound: 100 + 1.5 * 1.0 = 101.5
    Lower bound: 100 - 1.0 * 1.0 = 99.0
    Time limit: 20 candles
    """
    dates = pd.date_range("2023-01-01 00:00", periods=50, freq="15min")
    df = pd.DataFrame({
        'time': dates,
        'open': 100.0,
        'high': 100.0,
        'low': 100.0,
        'close': 100.0,
        'atr_14': 1.0,  # Changed from ATRr_14
        'volume': 1000
    })
    
    for col in config.FEATURE_COLS if hasattr(config, 'FEATURE_COLS') else ['fake_feat1', 'fake_feat2']:
        df[col] = np.random.randn(len(df))
        
    return df

def test_apply_triple_barrier_hits_upper(synthetic_data):
    # Setup row 0 to hit upper barrier at row 5
    synthetic_data.loc[5, 'close'] = 102.0  # Above 101.5
    
    result = apply_triple_barrier(synthetic_data)
    assert 'label' in result.columns
    assert len(result) == len(synthetic_data) - config.TRIPLE_BARRIER_TIME_LIMIT
    # For row 0, it should hit upper (1)
    assert result.loc[0, 'label'] == 1

def test_apply_triple_barrier_hits_lower(synthetic_data):
    # Setup row 0 to hit lower barrier at row 5
    synthetic_data.loc[5, 'close'] = 98.0  # Below 99.0
    
    result = apply_triple_barrier(synthetic_data)
    assert result.loc[0, 'label'] == -1

def test_apply_triple_barrier_hits_neither(synthetic_data):
    # Prices stay at 100.0, so neither bound is hit within 20 candles
    result = apply_triple_barrier(synthetic_data)
    assert result.loc[0, 'label'] == 0

def test_get_label_distribution(caplog):
    df = pd.DataFrame({'label': [-1, -1, 0, 1, 1, 1]})
    dist = get_label_distribution(df)
    
    assert dist[-1]['count'] == 2
    assert dist[0]['count'] == 1
    assert dist[1]['count'] == 3
    assert dist[1]['percentage'] == 50.0

def test_prepare_training_data(monkeypatch):
    import features
    monkeypatch.setattr(features, 'FEATURE_COLS', ['feat_1'])
    
    labels = [0] * 50 + [1] * 2 + [-1] * 1
    df = pd.DataFrame({
        'feat_1': np.random.randn(len(labels)),
        'label': labels
    })
    
    X, y_encoded, le = prepare_training_data(df)
    
    # Classes were: -1, 0, 1 -> le classes: 0, 1, 2
    # The new majority class 0 (encoded as 1) should be downsampled
    # Others: count = 3. Target 0 count = others * 10 = 30.
    # Total expected len = 30 + 3 = 33
    assert len(y_encoded) == 33
    assert list(le.classes_) == [-1, 0, 1]

def test_save_prepared_data(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / "data")
    monkeypatch.setattr(config, 'MODELS_DIR', tmp_path / "models")
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    monkeypatch.setattr(config, 'ENCODER_PATH', str(config.MODELS_DIR / "label_encoder.pkl"))
    
    X = pd.DataFrame({'f1': [1,2,3]})
    y = np.array([0,1,2])
    le = LabelEncoder()
    le.fit([-1,0,1])
    
    save_prepared_data(X, y, le)
    
    assert os.path.exists(config.DATA_DIR / "X_train.csv")
    assert os.path.exists(config.DATA_DIR / "y_train.npy")
    assert os.path.exists(config.ENCODER_PATH)
