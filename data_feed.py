"""
data_feed.py

Handles connecting to MetaTrader 5 (MT5), fetching historical market data,
and retrieving real-time ticks.
"""

import os
import logging
from typing import Optional, Dict, Any
import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv
import config

# Setup logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def connect_mt5() -> bool:
    """
    Connects to the MetaTrader 5 terminal using credentials from .env.
    
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    path = os.getenv("MT5_PATH")
    server = os.getenv("MT5_SERVER")
    login_str = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")

    login = int(login_str) if login_str else 0

    try:
        if path:
            initialized = mt5.initialize(path=path)
        else:
            initialized = mt5.initialize()

        if not initialized:
            logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
            return False

        if login and password and server:
            authorized = mt5.login(login=login, password=password, server=server)
            if not authorized:
                logger.error(f"Failed to connect to MT5 account {login}: {mt5.last_error()}")
                return False
            
        logger.info(f"Successfully connected to MT5 account {login}")
        return True
    except Exception as e:
        logger.error(f"Exception during MT5 connection: {str(e)}")
        return False

def disconnect_mt5() -> None:
    """
    Disconnects from the MetaTrader 5 terminal.
    """
    try:
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    except Exception as e:
        logger.error(f"Exception during MT5 shutdown: {str(e)}")

def get_historical_data(symbol: str, timeframe: int, num_candles: int) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data from MT5.

    Args:
        symbol (str): The trading symbol.
        timeframe (int): The MT5 timeframe constant.
        num_candles (int): The number of candles to fetch.

    Returns:
        Optional[pd.DataFrame]: DataFrame with historical data or None on failure.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch historical data for {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        logger.info(f"Fetched {len(df)} historical candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Exception during historical data fetch for {symbol}: {str(e)}")
        return None

def get_current_tick(symbol: str) -> Optional[Dict[str, float]]:
    """
    Gets the latest tick (bid/ask) for a given symbol.

    Args:
        symbol (str): The trading symbol.

    Returns:
        Optional[Dict[str, float]]: Dictionary with 'bid' and 'ask' prices, or None on failure.
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to fetch tick for {symbol}: {mt5.last_error()}")
            return None
        
        return {'bid': tick.bid, 'ask': tick.ask, 'time': float(tick.time)}
    except Exception as e:
        logger.error(f"Exception during tick fetch for {symbol}: {str(e)}")
        return None

def get_account_info() -> Optional[Dict[str, Any]]:
    """
    Retrieves current account information.

    Returns:
        Optional[Dict[str, Any]]: Account info dictionary or None on failure.
    """
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None
            
        return account_info._asdict()
    except Exception as e:
        logger.error(f"Exception fetching account info: {str(e)}")
        return None
