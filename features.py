"""
features.py

Feature engineering module. Uses pandas-ta to compute technical indicators 
on OHLCV data fetched from MT5.
"""

import logging
import pandas as pd
import pandas_ta as ta
import config

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of technical indicators to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'open', 'high', 'low', 'close', 'tick_volume'.
        
    Returns:
        pd.DataFrame: DataFrame with new indicator columns.
    """
    try:
        if df.empty or len(df) < 50:
            logger.warning("DataFrame is too small to calculate technical indicators properly.")
            return df

        # We need a copy to avoid SettingWithCopyWarning
        df_ind = df.copy()
        
        # Mapping tick_volume to volume for pandas-ta
        if 'tick_volume' in df_ind.columns and 'volume' not in df_ind.columns:
            df_ind['volume'] = df_ind['tick_volume']
            
        # 1. Moving Averages
        df_ind.ta.ema(length=9, append=True)
        df_ind.ta.ema(length=21, append=True)
        df_ind.ta.sma(length=50, append=True)
        df_ind.ta.sma(length=200, append=True)
        
        # 2. Momentum
        df_ind.ta.rsi(length=14, append=True)
        df_ind.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # 3. Volatility
        df_ind.ta.atr(length=config.ATR_PERIOD, append=True)
        df_ind.ta.bbands(length=20, std=2, append=True)
        
        # 4. Trend
        df_ind.ta.adx(length=14, append=True)
        
        # Drop rows with NaN values created by window functions
        df_ind.dropna(inplace=True)
        
        logger.info(f"Successfully added technical indicators. Shape: {df_ind.shape}")
        return df_ind

    except Exception as e:
        logger.error(f"Exception during feature engineering: {str(e)}")
        # Return original DataFrame on failure
        return df
