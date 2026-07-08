"""
config.py

Centralized configuration file for the Forex rules-based bot.
Contains broker selection, The5ers Bootcamp risk parameters, and paths.
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
BOT_LABEL = "RULES_BOT_V2"

# cTrader Open API
CTRADER_ENV = os.getenv("CTRADER_ENV", "demo").lower()
CTRADER_UNITS_PER_LOT = 100_000

# ---------------------------------------------------------------------------
# The5ers BOOTCAMP rules ($20K plan: steps $5k → $10k → $15k)
#   Official per step: target +6%; max loss −5% STATIC from the initial step
#   balance; no official daily pause during challenge steps; 1:30 leverage;
#   unlimited time; 30-day inactivity closure; every order needs a visible SL;
#   no bulk trading.
#
# The official −5% limit is NEVER the bot's working limit: the bot flattens
# and disables itself at −3% (KILL_SWITCH_PCT) so a slippage/gap on the last
# trade cannot reach the official breach level.
# ---------------------------------------------------------------------------
STARTING_BALANCE = None
MAX_DRAWDOWN_LIMIT = 0.05             # Official Bootcamp max loss (static, informational)
KILL_SWITCH_PCT = 0.03                # Operative hard stop: flatten + disable (persisted)
DRAWDOWN_WARNING_THRESHOLD = 0.02     # Soft threshold → reduce position size 50%
MAX_DRAWDOWN_PCT = MAX_DRAWDOWN_LIMIT  # Backward-compat alias (legacy refs + tests)

# Self-imposed pacing stops (Bootcamp has no official daily pause during steps)
DAILY_LOSS_PCT = 0.0075               # Stop for the day at −0.75% from day-start balance
WEEKLY_STOP_PCT = 0.015               # Stop for the week at −1.5% from week-start balance
MAX_CONSEC_LOSSES_DAY = 2             # Stop for the day after 2 consecutive losses
MAX_CONSEC_LOSSES_WEEK = 5            # Stop for the week after 5 consecutive losses
MAX_TRADES_PER_DAY = 2                # Hard cap on entries per server day

RISK_PER_TRADE_PCT = 0.003            # 0.3% per trade; safety over speed

# Stop Loss / Take Profit
SL_ATR_MULT = 1.5                     # SL = 1.5×ATR beyond the pullback swing
TP_R = 2.0                            # TP in R multiples of the ACTUAL SL distance
BE_AT_R = 1.0                         # Move SL to breakeven at +1.0R (once, rate-limited)
SL_MIN_PIPS = 8.0                     # Skip trade if clamp violated (never widen/narrow)
SL_MAX_PIPS = 25.0

# Entry filters
MAX_SPREAD_PIPS = 1.2                 # Skip entries when spread exceeds this
ATR_MIN_PIPS = 4.0                    # Skip if ATR(14, M15) below this
ATR_MAX_MEDIAN_MULT = 3.0             # Skip if ATR > 3× its 20-day median
USE_ADX_GATE = True
ADX_MIN = 20.0                        # H1 ADX(14) regime gate
ADX_MIN_RELAXED = 15.0                # Heartbeat mode relaxes the gate one notch

# Step target / inactivity heartbeat
PROFIT_TARGET_PCT = 0.06              # Bootcamp step target (+6%)
HEARTBEAT_DAYS = 21                   # If no trade in 21 days, permit one reduced-risk
HEARTBEAT_RISK_PCT = 0.001            # ... setup at 0.1% risk (avoids 30-day closure)

# Meta-model veto (XGBoost). It may only BLOCK candidates, never create them.
USE_META_VETO = False
META_VETO_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.65                 # Legacy ML-entry threshold (entries retired; kept
                                      # for the deprecated predict_signal helper only)

# Execution Limits
MAX_OPEN_TRADES = 1                   # The5ers prohibits bulk trading
MIN_TRADE_DURATION = 60
NEWS_BUFFER_MINS = 30                 # High-impact EUR/USD events: no entries ±30 min
NEWS_MAJOR_BUFFER_MINS = 60           # NFP/US CPI/FOMC/ECB: no entries ±60 min
NEWS_FLATTEN_BEFORE_MINS = 15         # ... and flatten open positions 15 min before
BE_MODIFY_MIN_INTERVAL_S = 60         # Rate limit on SL modifications (no EA spam)
ROLLOVER_START_UTC = 21
ROLLOVER_END_UTC = 22
BOT_MAGIC_NUMBER = 123456

# Trading Sessions
# Entry window is defined in LONDON LOCAL time (handles DST via zoneinfo);
# the fixed-UTC constants below remain only for legacy feature engineering.
LONDON_TZ = "Europe/London"
SESSION_START_LONDON = 8              # Entries allowed 08:00–17:00 Europe/London
SESSION_END_LONDON = 17
FRIDAY_CUTOFF_LONDON = 15             # No entries Friday after 15:00 London
SUNDAY_OPEN_BLOCK_HOURS = 2           # No entries first 2h after Sunday open
# Server-time no-trade window around rollover/day boundary (HH, MM) → (HH, MM)
SERVER_TZ = "UTC"                     # Set to the broker server tz (e.g. "Etc/GMT-3")
NO_TRADE_SERVER_WINDOW = ((21, 45), (0, 15))

# Legacy fixed-UTC session constants (feature engineering only — not entry gating)
LONDON_START_UTC = 7
LONDON_END_UTC = 16
NY_START_UTC = 13
NY_END_UTC = 21

# Triple Barrier Labelling Parameters (research pipeline)
TRIPLE_BARRIER_UPPER_MULT = 0.9
TRIPLE_BARRIER_LOWER_MULT = 0.6
TRIPLE_BARRIER_TIME_LIMIT = 25

# Backtest cost model (conservative; applied to the rules strategy)
BACKTEST_SPREAD_FLOOR_PIPS = 0.6      # Recorded/estimated spread floored here
BACKTEST_COMMISSION_PER_LOT_RT = 7.0  # USD per standard lot, round trip
BACKTEST_SLIPPAGE_ENTRY_PIPS = 0.3
BACKTEST_SLIPPAGE_STOP_PIPS = 1.0
BACKTEST_SLIPPAGE_NEWS_PIPS = 2.0

# Paths
MODEL_PATH = str(MODELS_DIR / "model.pkl")
ENCODER_PATH = str(MODELS_DIR / "label_encoder.pkl")
LOG_PATH = str(LOGS_DIR / "bot.log")
JOURNAL_PATH = "trades_log.csv"
RISK_DISABLED_FLAG_PATH = str(LOGS_DIR / "trading_disabled.json")
RISK_STATE_PATH = str(LOGS_DIR / "risk_state.json")
