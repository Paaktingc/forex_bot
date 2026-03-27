"""
labelling.py

Generates target variables for the ML models by simulating 
whether a trade hits TP before SL based on future price action.
"""

import logging
import pandas as pd
import numpy as np
import config

logger = logging.getLogger(__name__)

def create_labels(df: pd.DataFrame, lookforward: int = 100) -> pd.DataFrame:
    """
    Creates target labels for ML training.
    Logic:
      If long TP is hit before long SL -> Class 1 (Buy)
      If short TP is hit before short SL -> Class 2 (Sell)
      Otherwise -> Class 0 (Hold/Loss)
      
    Args:
        df (pd.DataFrame): DataFrame with features, including 'ATR_14'.
        lookforward (int): Number of future candles to look ahead for TP/SL.
        
    Returns:
        pd.DataFrame: DataFrame with a new 'target' column.
    """
    try:
        if df.empty or len(df) <= lookforward:
            logger.warning("DataFrame too small to create labels.")
            return df
            
        df_labels = df.copy()
        target = np.zeros(len(df_labels), dtype=int)
        
        # We need ATR for TP/SL calculations. Fallback if not found.
        atr_col = [c for c in df_labels.columns if 'ATR' in c]
        if not atr_col:
            logger.error("ATR column not found. Run features.py first.")
            return df_labels
        
        atr_col_name = atr_col[0]
        
        closes = df_labels['close'].values
        highs = df_labels['high'].values
        lows = df_labels['low'].values
        atrs = df_labels[atr_col_name].values
        
        for i in range(len(df_labels) - lookforward):
            entry_price = closes[i]
            atr = atrs[i]
            
            if pd.isna(atr) or atr <= 0:
                continue
                
            long_tp = entry_price + (atr * config.TP_ATR_MULTIPLIER)
            long_sl = entry_price - (atr * config.SL_ATR_MULTIPLIER)
            
            short_tp = entry_price - (atr * config.TP_ATR_MULTIPLIER)
            short_sl = entry_price + (atr * config.SL_ATR_MULTIPLIER)
            
            long_outcome = 0
            short_outcome = 0
            
            # Forward look
            for j in range(i + 1, i + lookforward):
                future_high = highs[j]
                future_low = lows[j]
                
                # Check Long
                if long_outcome == 0:
                    if future_low <= long_sl:
                        long_outcome = -1 # SL hit
                    elif future_high >= long_tp:
                        long_outcome = 1 # TP hit
                        
                # Check Short
                if short_outcome == 0:
                    if future_high >= short_sl:
                        short_outcome = -1 # SL hit
                    elif future_low <= short_tp:
                        short_outcome = 1 # TP hit
                        
                if long_outcome != 0 and short_outcome != 0:
                    break
                    
            if long_outcome == 1 and short_outcome <= 0:
                target[i] = 1
            elif short_outcome == 1 and long_outcome <= 0:
                target[i] = 2
                
        df_labels['target'] = target
        
        # Drop the last 'lookforward' rows as their targets are unknown/invalid
        df_labels = df_labels.iloc[:-lookforward].copy()
        
        logger.info(f"Successfully created labels. Buy: {(target==1).sum()}, Sell: {(target==2).sum()}, Hold: {(target==0).sum()}")
        return df_labels
        
    except Exception as e:
        logger.error(f"Exception during target labelling: {str(e)}")
        return df
