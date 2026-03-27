"""
data_feed.py
"""

import os
import logging
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

import config

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available — Mac/dev mode")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _require_mt5() -> None:
    if not MT5_AVAILABLE:
        raise RuntimeError(
            "MT5 not available. Deploy to Windows VPS for live trading."
        )

def connect_mt5() -> float:
    """Connects to MT5 and returns account balance."""
    try:
        _require_mt5()
        load_dotenv()
        login_str = os.getenv("MT5_LOGIN") or os.getenv("MT5_ACCOUNT")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        
        if not (login_str and password and server):
            raise ConnectionError("Missing MT5 credentials in .env")
            
        login = int(login_str)
        
        if not mt5.initialize(login=login, password=password, server=server):
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
            
        account_info = mt5.account_info()
        if account_info is None:
            raise ConnectionError(f"Failed to get account info: {mt5.last_error()}")
            
        balance = float(account_info.balance)
        logger.info(f"Connected | Account: {login} | Balance: {balance}")
        
        return balance
        
    except Exception as e:
        logger.error(f"Error connecting to MT5: {str(e)}")
        raise

def get_ohlcv(symbol: str, timeframe_str: str, bars: int) -> pd.DataFrame:
    """Fetches historical OHLCV data."""
    try:
        _require_mt5()
        timeframe_map = {
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1
        }
        
        if timeframe_str not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}")
            
        timeframe = timeframe_map[timeframe_str]
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch rates for {symbol}: {mt5.last_error()}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        
        # Convert time to datetime and make timezone-aware UTC
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        # Keep open, high, low, close, volume (mapped from tick_volume)
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        # Filter only existing columns just in case
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        df = df[columns_to_keep]
        
        # Drop rows with NaN or zero volume
        df.dropna(inplace=True)
        if 'volume' in df.columns:
            df = df[df['volume'] > 0]
            
        logger.info(f"Fetched {len(df)} bars for {symbol} on {timeframe_str}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {str(e)}")
        raise

def get_ohlcv_from_csv(symbol: str, timeframe_str: str) -> pd.DataFrame:
    """Loads OHLCV data from project CSV files for offline/backtest use."""
    filepath = config.DATA_DIR / f"{symbol}_{timeframe_str}_real.csv"
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        columns_to_keep = ["open", "high", "low", "close", "volume"]
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        df = df[columns_to_keep].dropna()
        logger.info(f"Loaded {len(df)} CSV bars for {symbol} on {timeframe_str}")
        return df
    except Exception as e:
        logger.error(f"Error loading OHLCV CSV data from {filepath}: {str(e)}")
        raise

def get_latest_tick(symbol: str) -> Dict[str, float]:
    """Retrieves the latest tick and calculates spread in pips."""
    try:
        _require_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Failed to fetch tick for {symbol}: {mt5.last_error()}")
            
        ask = float(tick.ask)
        bid = float(tick.bid)
        spread_pips = (ask - bid) / 0.00001
        
        logger.info(f"Fetched tick for {symbol}: Bid={bid}, Ask={ask}, Spread(pips)={spread_pips:.1f}")
        return {
            "ask": ask,
            "bid": bid,
            "spread": spread_pips
        }
        
    except Exception as e:
        logger.error(f"Error fetching latest tick: {str(e)}")
        raise

def get_account_info() -> Dict[str, float]:
    """Retrieves account information including calculated drawdown percentage."""
    try:
        _require_mt5()
        info = mt5.account_info()
        if info is None:
            raise ValueError(f"Failed to fetch account info: {mt5.last_error()}")
            
        balance = float(info.balance)
        equity = float(info.equity)
        margin = float(info.margin)
        free_margin = float(info.margin_free)
        
        drawdown_pct = ((balance - equity) / balance * 100.0) if balance > 0 else 0.0
        
        logger.info(f"Account Info: Balance={balance}, Equity={equity}, Drawdown={drawdown_pct:.2f}%")
        return {
            "balance": balance,
            "equity": equity,
            "margin": margin,
            "free_margin": free_margin,
            "drawdown_pct": drawdown_pct
        }
        
    except Exception as e:
        logger.error(f"Error fetching account info: {str(e)}")
        raise

def save_data(df: pd.DataFrame, filename: str) -> None:
    """Saves DataFrame to data directory."""
    try:
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", f"{filename}.csv")
        df.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        raise

def load_data(filename: str) -> pd.DataFrame:
    """Loads DataFrame from data directory."""
    try:
        filepath = os.path.join("data", f"{filename}.csv")
        df = pd.read_csv(filepath, index_col='time', parse_dates=True)
        # Ensure it's timezone-aware UTC if not already
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        raise
