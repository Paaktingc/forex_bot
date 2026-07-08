# forex_bot Repository Scope

## Purpose
A fully automated RULES-FIRST forex trading system built for EURUSD, targeted at The5ers Bootcamp ($20K plan; +6% per step, −5% static max loss, −3% self-imposed kill switch). The strategy is deterministic (`strategy.py`); the ML model survives only as an optional veto. Supports live execution (demo only) and research/backtesting, with broker abstraction for MetaTrader 5 (MT5) and cTrader Open API.

## Key Components

### Live Trading
- `main.py`
  - Live trading loop on 15-minute candles (dry-run supported)
  - Flow: strategy candidate → session/news/spread/volatility filters →
    optional MetaVeto (block-only) → RiskManager → execution
  - Per-candle position management: breakeven at +1R, flatten before major news
- `strategy.py`
  - Rules signal engine: H1 EMA50/EMA200 + ADX regime, M15 EMA20-pullback /
    H1-swing-retracement entry with RSI(14) 50-recross trigger
  - Spread / volatility / DST-aware London session filters

### Backtesting & Research
- `backtest.py`
  - `RulesBacktestEngine`: rules-strategy simulation with conservative costs
    and live pacing gates; `python backtest.py --go-no-go` prints the
    GO/NO-GO verdict (backtest + walk-forward + Monte Carlo)
  - Legacy model-driven `BacktestEngine` retained for research
    (`--legacy-model`)
- `run_pipeline.py`
  - Research/meta-labeling pipeline
  - Loads raw EURUSD data, resamples to H1, engineers features
  - Generates base signals, labels via triple barrier
  - Runs regime diagnostics, feature importance, lookahead audits, walk-forward validation

### Model Training / Inference
- `model.py`
  - `MetaVeto`: the XGBoost model demoted to a blocking-only filter on rules
    candidates (off by default via `USE_META_VETO`); it can never create trades
  - XGBoost classifier training with time-series cross-validation
  - Label encoder and optional class-weighting
  - Model persistence to `models/model.pkl`

### Data Handling
- `data_feed.py`
  - Live broker OHLCV and tick access
  - CSV loading for offline/backtest use
  - Account info retrieval and broker connection management
- `features.py`
  - Feature engineering for legacy live trading and meta-label research
  - Defines technical, regime, volatility, and time-based features
- `labelling.py`
  - Triple barrier labelling and base signal generation
  - Regime filtering and label distribution helpers

### Execution and Risk
- `execution.py`
  - Order placement and closing (market orders only; every order must carry
    a broker-visible SL — rejected here otherwise)
  - Rate-limited SL modification and once-per-ticket breakeven moves
  - Throttles orders and handles MT5/cTrader execution
- `risk_manager.py`
  - The5ers Bootcamp rules: −3% kill switch (flatten + disk-persisted
    disabled flag, manual re-arm), −5% official backstop, weekly stop
    (−1.5% / 5 consecutive losses), daily stop (−0.75% / 2 trades /
    2 consecutive losses), server-time resets, inactivity heartbeat
  - 0.3% risk sizing (lots floored to 0.01), SL 1.5×ATR beyond the pullback
    swing clamped to 8–25 pips (skip outside clamp), TP = 2R
- `news_filter.py`
  - Tiered news blackouts (±30 min high impact; ±60 min + flatten 15 min
    before NFP/US CPI/FOMC/ECB); fails closed when the calendar is
    unavailable or empty
- `monte_carlo_dd.py`
  - Bootcamp step simulator (+6% pass / −5% fail / −3% kill) and
    block-bootstrap Monte Carlo used by the GO/NO-GO verdict

### Broker Abstraction
- `brokers/`
  - `base.py`: broker adapter interface
  - `factory.py`: broker selection helper
  - `mt5_adapter.py`: MetaTrader 5 adapter
  - `ctrader_adapter.py`: cTrader Open API adapter

## Configuration
- `config.py`
  - Central settings: symbol, timeframes, risk parameters, broker mode, paths
- `.env.example` / `.env`
  - Environment variables for broker credentials and mode

## Reporting & Diagnostics
- `backtest_report.py`
  - Backtest report generation
- `feature_importance.py`
  - Permutation feature importance pipeline
- `regime_diagnostic.py`
  - Regime diagnostic analysis
- `validation.py`
  - Lookahead bias and validation helpers
- `report.py`
  - Report generation utilities

## Utilities & Support
- `prepare_data.py`
  - Data preparation helpers
- `resampler.py`
  - OHLCV resampling utilities
- `health_check.py`
  - Bot health verification
- `trade_journal.py`
  - Trade logging
- `walkthrough.md`
  - Project walkthrough and documentation

## Data
- `data/`
  - Live/offline data files: `EURUSD_M15_real.csv`, `EURUSD_H1_real.csv`
  - Feature and training artifacts: `X_train.csv`, `y_train.npy`
  - Raw HistData source files in `data/raw/`

## Tests
- `tests/`
  - Unit tests covering backtesting, data feed, execution, features, labelling, model, news filter, risk manager, trade journal, and main flow

## Usage Notes
- Designed for Python 3.11+
- Requires MT5 installed for MetaTrader live trading
- cTrader support requires `ctrader-open-api` credentials and SDK
- Strict risk management for The5ers-style evaluation
