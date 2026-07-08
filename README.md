# THE5ERS FOREX RULES BOT

A fully automated rules-first forex trading bot with selectable MetaTrader 5 (MT5) and cTrader Open API broker backends, targeted at The5ers **Bootcamp** ($20K plan: steps $5k → $10k → $15k; +6% target per step; −5% static max loss from the initial step balance; no official daily pause during steps; visible stop-loss required; no bulk trading).

The strategy IS the rules (H1 EMA regime + M15 pullback entries in `strategy.py`); the XGBoost model survives only as an optional veto (`MetaVeto`) that can block candidates but never create them (off by default). **Demo testing only.**

## Requirements
- Python 3.11+
- For MT5: Windows OS, MetaTrader 5 Terminal installed and running
- For cTrader: cTrader Open API app credentials and a demo or live cTrader account

## Setup Instructions

1. **Clone the Repository** and navigate to the project directory:
   ```bash
   cd forex_bot
   ```

2. **Create a Virtual Environment** and install dependencies:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy the `.env.example` file to `.env` and choose your broker:
   ```bash
   cp .env.example .env
   ```
   For MT5, keep `BROKER=mt5` and set `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER`.
   For cTrader demo, set `BROKER=ctrader`, `CTRADER_ENV=demo`, and fill in `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`, `CTRADER_ACCESS_TOKEN`, `CTRADER_REFRESH_TOKEN`, and `CTRADER_ACCOUNT_ID`.

4. **Verify Configurations**:
   Check `config.py` to reflect your specific phase thresholds, risk constraints, and parameters.

## Running the Bot

- **Testing**: Run `pytest tests/` to execute unit tests locally (a MacOS adapter/mock may be required if MT5 is absent).
- **Health Check**: Run `python health_check.py` after configuring `.env`.
- **Main Bot**: Run `python main.py` to start the live trading loop. For MT5, ensure MT5 is open or AutoLogin via the provided `.env`; for cTrader, start with a demo account.

## Risk rules (The5ers Bootcamp)

- Official Bootcamp max loss is **−5% static** from the initial step balance — but the bot's operative halt is a **−3% kill switch** (flatten everything, disable trading, persist a disabled flag that survives restarts and requires manual re-arm by deleting `logs/trading_disabled.json`).
- Self-imposed pacing (Bootcamp has no official daily pause during steps): stop for the day at **−0.75%**, 2 trades, or 2 consecutive losses; stop for the week at **−1.5%** or 5 consecutive losses (resets Monday, server time).
- 0.3% risk per trade, max 1 open trade, SL = 1.5×ATR beyond the pullback swing clamped to 8–25 pips (skip if violated), TP = 2R, breakeven at +1R.
- Entries only 08:00–17:00 Europe/London (DST-aware), never Friday after 15:00 London, in the first 2h after Sunday open, around rollover (21:45–00:15 server), when spread > 1.2 pips, or near high-impact news (±30 min; ±60 min and flatten 15 min before for NFP/US CPI/FOMC/ECB — fail closed if the calendar is unavailable).
- If no trade in 21 days, one setup is permitted at 0.1% risk with a relaxed ADX gate to avoid the 30-day inactivity closure.

## Backtest GO/NO-GO

```bash
python backtest.py --go-no-go
```

Runs the rules-strategy backtest with conservative costs (spread floored at 0.6 pips, $7/lot round-trip commission, slippage), a 24m/6m walk-forward with parameter perturbation, and ≥20,000 block-bootstrap Monte Carlo paths through a Bootcamp step simulator (absorb at +6% pass / −5% fail, −3% kill switch active). Prints GO only if P(breach −5%) < 1%, P(kill switch) < 10%, P(pass) > 70%, and PF ≥ 1.25 after costs in every fold.
