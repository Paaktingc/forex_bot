"""
labelling.py

Generates target variables for the ML models by using a triple-barrier 
labelling method. Also contains utilities to process data, prepare 
training sequences, handle class imbalance via downsampling, and persist data.
"""

import logging
import os
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder
from features import FEATURE_COLS

logger = logging.getLogger(__name__)

def apply_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the triple-barrier method to generate target labels.
    
    Barriers:
    1. Upper: entry + (TRIPLE_BARRIER_UPPER_MULT * ATR_14)
    2. Lower: entry - (TRIPLE_BARRIER_LOWER_MULT * ATR_14)
    3. Time: Next TRIPLE_BARRIER_TIME_LIMIT candles
    
    Logic:
      - First hit upper -> label 1 (Buy)
      - First hit lower -> label -1 (Sell)
      - Hit time limit or neither -> label 0 (No trade)
    
    Args:
        df: DataFrame containing features, 'close', and 'atr_14'.
        
    Returns:
        DataFrame with an added 'label' column.
    """
    df = df.copy()
    if df.empty or 'close' not in df.columns or 'atr_14' not in df.columns:
        logger.warning("Missing required columns ('close', 'atr_14') or DataFrame empty.")
        return df

    upper_mult = config.TRIPLE_BARRIER_UPPER_MULT
    lower_mult = config.TRIPLE_BARRIER_LOWER_MULT
    time_limit = config.TRIPLE_BARRIER_TIME_LIMIT
    
    labels = np.zeros(len(df), dtype=int)
    
    closes = df['close'].values
    atrs = df['atr_14'].values
    highs = df.get('high', df['close']).values
    lows = df.get('low', df['close']).values

    for i in range(len(df) - time_limit):
        entry = closes[i]
        atr = atrs[i]
        
        if pd.isna(atr) or atr <= 0:
            labels[i] = 0
            continue
            
        upper = entry + upper_mult * atr
        lower = entry - lower_mult * atr
        
        label = 0
        for j in range(i + 1, i + 1 + time_limit):
            # To be more precise, we check high/low for touches or just close. 
            # Prompt specifies: "first future close >= upper -> label = 1, first future close <= lower -> label = -1"
            # Using close as per the prompt instructions.
            future_close = closes[j]
            
            if future_close >= upper:
                label = 1
                break
            elif future_close <= lower:
                label = -1
                break
                
        labels[i] = label

    df['label'] = labels
    
    # Drop last TIME_LIMIT rows to avoid lookahead bias
    return df.iloc[:-time_limit].copy()

def get_label_distribution(df: pd.DataFrame) -> dict:
    """
    Returns and prints the count and percentage of -1, 0, 1 labels.
    """
    if 'label' not in df.columns:
        return {}
        
    counts = df['label'].value_counts().to_dict()
    total = len(df)
    
    distribution = {}
    for label_val in [-1, 0, 1]:
        count = counts.get(label_val, 0)
        pct = (count / total * 100) if total > 0 else 0
        distribution[label_val] = {'count': count, 'percentage': pct}
        
    logger.info("Label Distribution:")
    for lbl, stats in distribution.items():
        logger.info(f"Class {lbl}: {stats['count']} ({stats['percentage']:.2f}%)")
        
    return distribution

def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Extracts features and target, encodes labels, and downsamples 
    the majority class (0) if there's significant imbalance.
    
    Args:
        df: DataFrame containing FEATURE_COLS and 'label'.
        
    Returns:
        X (DataFrame): Features.
        y_encoded (np.ndarray): Target classes (0, 1, 2).
        le (LabelEncoder): Fitted encoder mapping original to new labels.
    """
    # Filter only available feature cols
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing FEATURE_COLS in df: {missing_cols}")
        
    valid_feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    # Drop rows with NaN in features or label
    df_clean = df.dropna(subset=valid_feature_cols + ['label']).copy()
    
    X = df_clean[valid_feature_cols]
    y = df_clean['label'].values
    
    # Downsample if class 0 is overly dominant (> 10:1 ratio)
    counts = pd.Series(y).value_counts()
    count_0 = counts.get(0, 0)
    count_others = max(1, len(y) - count_0)
    
    if count_0 / count_others > 10:
        logger.info("Significant class imbalance detected. Downsampling class 0.")
        target_0_count = count_others * 2  # Keep 2:1 ratio max against others, or simply 10:1 if strictly needed.
        # Downsample to 10:1 to keep within limits, or match majority closely.
        target_0_count = count_others * 10
        
        idx_0 = np.where(y == 0)[0]
        idx_others = np.where(y != 0)[0]
        
        np.random.seed(42)  # For reproducibility
        target_0_count = min(target_0_count, len(idx_0))
        idx_0_sampled = np.random.choice(idx_0, size=target_0_count, replace=False)
        
        selected_idx = np.sort(np.concatenate([idx_0_sampled, idx_others]))
        X = X.iloc[selected_idx]
        y = y[selected_idx]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def save_prepared_data(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder) -> None:
    """
    Saves features, target, and the label encoder to disk.
    
    Args:
        X: Feature dataframe.
        y: Encoded target array.
        le: Fitted LabelEncoder.
    """
    import pickle
    
    X.to_csv(config.DATA_DIR / "X_train.csv", index=False)
    np.save(config.DATA_DIR / "y_train.npy", y)
    
    with open(config.ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
        
    logger.info(f"Saved prepared data and encoder to {config.DATA_DIR} and {config.MODELS_DIR}")
