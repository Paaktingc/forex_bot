"""
test_labelling.py

Unit tests for the labelling.py module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from labelling import create_labels

def test_create_labels():
    rows = 150
    df = pd.DataFrame({
        'close': np.linspace(100, 110, rows),
        'high': np.linspace(100, 110, rows) + 1,
        'low': np.linspace(100, 110, rows) - 1,
        'ATR_14': np.ones(rows) * 1.0  # ATR=1
    })
    
    # In this upward trend scenario, long trades should hit TP before SL
    df_labelled = create_labels(df, lookforward=20)
    
    assert 'target' in df_labelled.columns
    assert len(df_labelled) == rows - 20
    assert (df_labelled['target'] == 1).any()  # Should have some long signals

def test_create_labels_no_atr():
    df = pd.DataFrame({
        'close': [1,2,3],
        'high': [1,2,3],
        'low': [1,2,3]
    })
    df_labelled = create_labels(df, lookforward=1)
    
    # Should not crash, just returns df without target
    assert 'target' not in df_labelled.columns
