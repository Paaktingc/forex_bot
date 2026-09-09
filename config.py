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

# ===========================================================================
# PROGRAMME PROFILES — The5ers evaluation targets (verified 2026-07-21)
# ===========================================================================
# The frozen London strategy is programme-agnostic. All barrier geometry,
# circuit breakers, pacing stops and risk sizing live in one profile so the
# SAME edge can target a different evaluation without touching strategy.py.
#
# Select with the PROGRAMME env var (default "bootcamp"): "bootcamp" | "high_stakes".
#
# Rule sources (first-party, dated):
#   Bootcamp    the5ers.com/bootcamp + help.the5ers.com (upd. 01.07.2026)
#   High Stakes the5ers.com/high-stakes + help center max-loss article
#
# Cycle-6 finding: the frozen edge models NO-GO on Bootcamp (P(pass)~66%,
# P(kill)~34%) but ~73–93% two-step completion on High Stakes, because a
# symmetric 10%/10% barrier with unlimited time suits a thin positive edge
# far better than an asymmetric +6/−3 gauntlet. See cycle6_research_report.md.
# ---------------------------------------------------------------------------
PROGRAMMES = {
    # 3-step Bootcamp: +6% target, −5% STATIC max loss per step; kill at −3%
    # so a gap/slippage on the last trade cannot reach the official breach.
    "bootcamp": {
        "label": "The5ers Bootcamp (3-step)",
        "steps": (0.06, 0.06, 0.06),   # per-step profit targets
        "max_drawdown_limit": 0.05,    # official static max loss per step
        "kill_switch_pct": 0.03,       # operative flatten+disable (< official)
        "drawdown_warning": 0.02,      # soft: halve position size
        "official_daily_loss": None,   # no official daily pause during steps
        "daily_loss_pct": 0.0075,      # self-imposed pacing stop
        "weekly_stop_pct": 0.015,
        "max_consec_losses_day": 2,
        "max_consec_losses_week": 5,
        "max_trades_per_day": 2,
        # 0.20%: the drawdown gate binds, not the kill. On the frozen EURUSD
        # ablation-(a) R distribution, P(maxDD>5%) over a 1y horizon is 2.0% at
        # 0.20% but 15.3% at the old 0.30% — three times over the repo's own
        # P(maxDD>5%)<5% hard gate (STEP 1 / GATE A, 2026-07-27). Feasible
        # window is risk ≤ 0.23%; 0.20% is chosen with margin. Enforced by
        # tests/test_programme_risk_gate.py.
        "risk_per_trade_pct": 0.002,
        "min_profitable_days": 0,      # none required
        "profitable_day_min_pct": 0.0,
    },
    # 2-step High Stakes: 10% then 5% target, −10% STATIC max loss, 5% daily
    # loss, ≥3 profitable days (a day with closed profit ≥0.5% of balance).
    # Circuit breakers retuned to the 10% budget: kill −6%, hard warn −8%.
    "high_stakes": {
        "label": "The5ers High Stakes (2-step)",
        "steps": (0.10, 0.05),         # step 1 = 10%, step 2 = 5%
        "max_drawdown_limit": 0.10,    # official static max loss
        "kill_switch_pct": 0.06,       # operative flatten+disable (< official)
        "drawdown_warning": 0.04,      # soft: halve position size
        "official_daily_loss": 0.05,   # official 5% daily loss limit
        "daily_loss_pct": 0.015,       # self-imposed pacing (well inside 5%)
        "weekly_stop_pct": 0.030,
        "max_consec_losses_day": 3,
        "max_consec_losses_week": 6,
        "max_trades_per_day": 2,
        # 0.20%: P(maxDD>5%) is a property of the return stream, not the
        # programme budget — at the old 0.40% it was 35.2%, far past the
        # P(maxDD>5%)<5% hard gate. The wider −10% High Stakes budget does not
        # license a higher per-trade risk on this thin edge; it buys completion
        # probability with time (STEP 1e). 0.20% → P(maxDD>5%)=2.0%.
        "risk_per_trade_pct": 0.002,
        "min_profitable_days": 3,      # ≥3 profitable days per step
        "profitable_day_min_pct": 0.005,  # a "profitable day" = closed +0.5%
    },
}

PROGRAMME = os.getenv("PROGRAMME", "bootcamp").lower()
if PROGRAMME not in PROGRAMMES:
    raise ValueError(
        f"Unknown PROGRAMME={PROGRAMME!r}; choose one of {list(PROGRAMMES)}"
    )
_P = PROGRAMMES[PROGRAMME]

# --- Derived programme constants (consumed by risk_manager / backtest / MC) ---
STARTING_BALANCE = None
PROGRAMME_STEPS = _P["steps"]                       # tuple of per-step targets
PROFIT_TARGET_PCT = _P["steps"][0]                  # first-step target (legacy alias)
MAX_DRAWDOWN_LIMIT = _P["max_drawdown_limit"]       # official static max loss
KILL_SWITCH_PCT = _P["kill_switch_pct"]             # operative hard stop
DRAWDOWN_WARNING_THRESHOLD = _P["drawdown_warning"] # soft size-reduction
MAX_DRAWDOWN_PCT = MAX_DRAWDOWN_LIMIT               # backward-compat alias
OFFICIAL_DAILY_LOSS_PCT = _P["official_daily_loss"] # None for Bootcamp steps

DAILY_LOSS_PCT = _P["daily_loss_pct"]               # self-imposed daily pacing
WEEKLY_STOP_PCT = _P["weekly_stop_pct"]
MAX_CONSEC_LOSSES_DAY = _P["max_consec_losses_day"]
MAX_CONSEC_LOSSES_WEEK = _P["max_consec_losses_week"]
MAX_TRADES_PER_DAY = _P["max_trades_per_day"]

RISK_PER_TRADE_PCT = _P["risk_per_trade_pct"]

# High Stakes profitable-day gate (0 / 0.0 disables it for Bootcamp)
MIN_PROFITABLE_DAYS = _P["min_profitable_days"]
PROFITABLE_DAY_MIN_PCT = _P["profitable_day_min_pct"]

# Stop Loss / Take Profit
SL_ATR_MULT = 1.5                     # SL = 1.5×ATR beyond the pullback swing
TP_R = 2.0                            # TP in R multiples of the ACTUAL SL distance
BE_AT_R = 1.0                         # Move SL to breakeven at +1.0R (once, rate-limited)
SL_MIN_PIPS = 8.0                     # Skip trade if clamp violated (never widen/narrow)
SL_MAX_PIPS = 25.0

# Entry generation
# "pullback_rsi": original M15 pullback + RSI recross (measured ~PF 0.99 —
#   the confirmation trigger subtracts the regime edge; see research_log.md)
# "regime_daily": ONE entry per London day at the first in-session bar where
#   the H1 regime held at the prior close; SL anchored at entry. Best honest
#   configuration found (PF 1.10 on 2015-2025 design window) but still
#   NO-GO against the Bootcamp gates — do not deploy without a new edge.
ENTRY_MODE = "regime_daily"

# Entry filters
MAX_SPREAD_PIPS = 1.2                 # Skip entries when spread exceeds this
ATR_MIN_PIPS = 4.0                    # Skip if ATR(14, M15) below this
ATR_MAX_MEDIAN_MULT = 3.0             # Skip if ATR > 3× its 20-day median
USE_ADX_GATE = True
ADX_MIN = 20.0                        # H1 ADX(14) regime gate
ADX_MIN_RELAXED = 15.0                # Heartbeat mode relaxes the gate one notch

# Step target / inactivity heartbeat
# (PROFIT_TARGET_PCT is now derived from the active PROGRAMME profile above.)
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
# Timezone in which the Forex Factory calendar is RENDERED for the scraper.
# FF localizes event times to the account/cookie timezone; parsing them as UTC
# blindly can shift every blackout by hours (Finding 3). Set this to the IANA
# zone your FF session renders in (verify once against a known event time), or
# set your FF account to GMT and leave "UTC". Times are localized to this zone
# then converted to UTC.
NEWS_SOURCE_TZ = os.getenv("NEWS_SOURCE_TZ", "UTC")

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

# Backtest cost model — VERIFIED against The5ers Help Center ("What are the
# spreads and commissions?", updated 02.01.2026): majors 0.2–0.9 pips,
# forex commission $4/lot round trip. Floors sit at the upper-middle of the
# quoted spread range; slippage stays our own conservative assumption.
BACKTEST_SPREAD_FLOOR_PIPS = 0.4      # Recorded/estimated spread floored here
# Per-symbol spread floors (pips)
BACKTEST_SPREAD_FLOOR_BY_SYMBOL = {
    "EURUSD": 0.4,
    "GBPUSD": 0.6,
    "AUDUSD": 0.6,
    "USDJPY": 0.5,
    # non-FX floors are ASSUMED pending MT5 spec freeze (cycle 5):
    "XAUUSD": 2.5,   # $0.25 (pip = 0.1)
    "GRXEUR": 1.5,   # 1.5 index points (pip = 1.0)
    "ETXEUR": 1.5,   # 1.5 index points (pip = 1.0)
}
BACKTEST_COMMISSION_PER_LOT_RT = 4.0  # USD per standard lot, round trip (verified)
BACKTEST_SLIPPAGE_ENTRY_PIPS = 0.3
BACKTEST_SLIPPAGE_STOP_PIPS = 1.0
BACKTEST_SLIPPAGE_NEWS_PIPS = 2.0

# ---------------------------------------------------------------------------
# Non-FX transfer research (research_log.md cycle 5).
# Pip-denominated thresholds convert ONCE to fractions of price, calibrated
# from EURUSD @ 1.10 (8 pips = 0.0008/1.10 etc.), applied at each trade's
# entry price for symbols whose asset_class != "fx". FX behavior unchanged.
# ---------------------------------------------------------------------------
BP_THRESHOLDS = {
    "sl_min": 0.0008 / 1.10,        # 7.27 bp of price  (8 pips)
    "sl_max": 0.0025 / 1.10,        # 22.73 bp          (25 pips)
    "atr_min": 0.0004 / 1.10,       # 3.64 bp           (4 pips)
    "slip_entry": 0.00003 / 1.10,   # 0.273 bp          (0.3 pips)
    "slip_stop": 0.0001 / 1.10,     # 0.909 bp          (1.0 pips)
    "slip_news": 0.0002 / 1.10,     # 1.818 bp          (2.0 pips)
}
# Index spread floors in points / XAU in pips-of-0.1 — ASSUMED pending a
# freeze against live The5ers MT5 symbol specifications. Indices carry no
# commission at The5ers; metals use a percentage commission (rate ASSUMED
# 0.002%/side pending MT5 spec).
BACKTEST_COMMISSION_PCT_PER_SIDE_BY_SYMBOL = {
    "XAUUSD": 0.00002,
}
CONTRACT_SIZE_BY_SYMBOL = {
    "XAUUSD": 100,        # oz per lot
    "GRXEUR": 1,          # 1 index unit per lot (nominal; cancels in R math)
    "ETXEUR": 1,
}
NEWS_CURRENCIES_BY_SYMBOL = {
    # European indices react to both ECB and the US majors calendar
    "GRXEUR": ("EUR", "USD"),
    "ETXEUR": ("EUR", "USD"),
    "XAUUSD": ("USD",),
}

# Paths
MODEL_PATH = str(MODELS_DIR / "model.pkl")
ENCODER_PATH = str(MODELS_DIR / "label_encoder.pkl")
LOG_PATH = str(LOGS_DIR / "bot.log")
JOURNAL_PATH = "trades_log.csv"
RISK_DISABLED_FLAG_PATH = str(LOGS_DIR / "trading_disabled.json")
RISK_STATE_PATH = str(LOGS_DIR / "risk_state.json")
