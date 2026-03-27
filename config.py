"""
config.py

Centralized configuration file for the MT5 Forex ML bot.
Contains all constants, risk parameters, 5ers rules, and model settings.
Never hardcode these values in other modules.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Trading Setup
SYMBOL = "EURUSD"
PRIMARY_TIMEFRAME = "M15"
TREND_TIMEFRAME = "H1"

# Risk Management & The5ers Hard Rules
PROFIT_TARGET_PCT = 0.06           # 6% per phase
MAX_DRAWDOWN_ABSOLUTE_PCT = 0.05   # 5% absolute from starting balance
DAILY_LOSS_LIMIT_PCT = 0.05        # 5% daily loss limit
DAILY_LOSS_KILL_SWITCH_PCT = 0.04  # Halt trading at 4% daily loss
DRAWDOWN_KILL_SWITCH_PCT = 0.045   # Halt trading at 4.5% absolute drawdown
RISK_PER_TRADE_PCT = 0.0075        # 0.75% of account equity per trade
MAX_CONCURRENT_TRADES = 2          # Maximum 2 concurrent open trades

# Stop Loss / Take Profit (ATR Multipliers)
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 1.0
TP_ATR_MULTIPLIER = 1.5
MIN_RR_RATIO = 1.5                 # 1:1.5 minimum

# Execution Limits
MIN_TRADE_DURATION_SECONDS = 60    # NO HFT: Minimum 60s trade duration
ORDER_DELAY_SECONDS = 2            # NO bulk orders: 2s delay between entries
ROLLOVER_START_UTC = "21:00"       # Rollover window start UTC
ROLLOVER_END_UTC = "22:00"         # Rollover window end UTC

HIGH_IMPACT_NEWS_BLOCK_MINUTES = 30 # No trading 30 mins before/after high impact news

# ML Model Parameters
CONFIDENCE_THRESHOLD = 0.65        # Minimum signal confidence for entry
STAGE_1_MODEL_NAME = "xgboost_model.json"
STAGE_2_MODEL_NAME = "lstm_model.h5"

# Logging configuration
LOG_FILE = LOGS_DIR / "trading_bot.log"
