"""
backtest.py

Standalone backtesting script to validate feature engineering, target generation,
and the ML model using historical data from MT5.
"""

import logging
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5
import config
from data_feed import connect_mt5, disconnect_mt5, get_historical_data
from features import add_technical_indicators
from labelling import create_labels
from model import Stage1Model

# Setup basic logging to console for backtesting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(symbol: str = config.SYMBOL, timeframe: int = 15, num_candles: int = 10000) -> dict:
    """
    Runs a vectorized backtest for the specified symbol.
    
    Args:
        symbol (str): Trading symbol.
        timeframe (int): Timeframe in minutes (e.g., 15 for M15).
        num_candles (int): Number of historical candles to fetch.
        
    Returns:
        dict: Backtest results.
    """
    logger.info(f"Starting backtest for {symbol} with {num_candles} candles.")
    
    # 1. Fetch Data
    if not connect_mt5():
        logger.error("Failed to connect to MT5 for backtesting.")
        return {}
        
    tf_constant = mt5.TIMEFRAME_M15 if timeframe == 15 else mt5.TIMEFRAME_H1
    
    df = get_historical_data(symbol, tf_constant, num_candles)
    disconnect_mt5()
    
    if df is None or df.empty:
        logger.error("No data fetched.")
        return {}
        
    # 2. Features
    logger.info("Computing features...")
    df = add_technical_indicators(df)
    
    # 3. Labels
    logger.info("Generating labels...")
    # Lookforward roughly 1 day for M15 (24 * 4 = 96)
    df = create_labels(df, lookforward=96)
    
    if 'target' not in df.columns or df.empty:
        logger.error("Labelling failed or data insufficient.")
        return {}
        
    # 4. Train Model
    # Split chronologically: 80% train, 20% test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    if train_df.empty or test_df.empty:
        logger.error("Insufficient data for train/test split after processing.")
        return {}
    
    model = Stage1Model()
    logger.info("Training Stage1 XGBoost Model...")
    model.train(train_df, target_col='target')
    
    # 5. Evaluate on test set
    features_only = test_df[model.features]
    preds = model.model.predict(features_only)
    actuals = test_df['target'].values
    
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(actuals, preds)
    
    logger.info(f"Backtest Test Set Accuracy: {acc:.4f}")
    
    return {
        'accuracy': acc,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }

if __name__ == "__main__":
    run_backtest()
