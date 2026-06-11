"""
config.py

Centralized configuration file for the Forex ML bot.
Contains broker selection, risk parameters, and model settings.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Trading Setup
SYMBOL = "EURUSD"
TIMEFRAME_PRIMARY = "M15"
TIMEFRAME_TREND = "H1"
BARS_TO_FETCH = 35000
BROKER = os.getenv("BROKER", "mt5").lower()
BOT_LABEL = "ML_BOT_V1"

# cTrader Open API
CTRADER_ENV = os.getenv("CTRADER_ENV", "demo").lower()
CTRADER_UNITS_PER_LOT = 100_000

# Risk Management & The5ers Hard Rules
STARTING_BALANCE = None
MAX_DRAWDOWN_LIMIT = 0.045            # Hard drawdown limit → halt all new trades
DRAWDOWN_WARNING_THRESHOLD = 0.030   # Soft threshold → reduce position size 50%
MAX_DRAWDOWN_PCT = MAX_DRAWDOWN_LIMIT  # Backward-compat alias (legacy refs + tests)
DAILY_LOSS_PCT = 0.040
RISK_PER_TRADE_PCT = 0.0039          # was 0.0075; ×0.52 to target ~4% max drawdown

# Stop Loss / Take Profit (ATR Multipliers)
SL_ATR_MULT = 1.0
TP_ATR_MULT = 1.5
MIN_CONFIDENCE = 0.65

# Execution Limits
MAX_OPEN_TRADES = 2
MIN_TRADE_DURATION = 60
NEWS_BUFFER_MINS = 30
ROLLOVER_START_UTC = 21
ROLLOVER_END_UTC = 22
BOT_MAGIC_NUMBER = 123456

# Trading Sessions
LONDON_START_UTC = 7
LONDON_END_UTC = 16
NY_START_UTC = 13
NY_END_UTC = 21

# Triple Barrier Labelling Parameters
TRIPLE_BARRIER_UPPER_MULT = 0.9
TRIPLE_BARRIER_LOWER_MULT = 0.6
TRIPLE_BARRIER_TIME_LIMIT = 25

# Paths
MODEL_PATH = str(MODELS_DIR / "model.pkl")
ENCODER_PATH = str(MODELS_DIR / "label_encoder.pkl")
LOG_PATH = str(LOGS_DIR / "bot.log")
JOURNAL_PATH = "trades_log.csv"
