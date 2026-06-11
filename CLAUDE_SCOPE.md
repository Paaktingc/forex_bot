# forex_bot Repository Scope

## Purpose
A fully automated Forex machine learning trading system built for EURUSD. It supports both live execution and research/backtesting, with broker abstraction for MetaTrader 5 (MT5) and cTrader Open API.

## Key Components

### Live Trading
- `main.py`
  - Live trading loop on 15-minute candles
  - Loads model and label encoder
  - Connects to configured broker
  - Applies risk and news filters
  - Generates live features and predicts a buy/sell/hold signal
  - Sizes risk, calculates SL/TP, and places orders
  - Supports dry-run mode

### Backtesting & Research
- `backtest.py`
  - Historical backtesting engine
  - Uses M15/H1 data, feature matrices, model inference, trade simulation
  - Computes performance metrics and The5ers-style pass/fail criteria
- `run_pipeline.py`
  - Research/meta-labeling pipeline
  - Loads raw EURUSD data, resamples to H1, engineers features
  - Generates base signals, labels via triple barrier
  - Runs regime diagnostics, feature importance, lookahead audits, walk-forward validation

### Model Training / Inference
- `model.py`
  - XGBoost classifier training with time-series cross-validation
  - Label encoder and optional class-weighting
  - Model persistence to `models/model.pkl`
  - Live prediction helper functions

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
  - Order placement and closing
  - Throttles orders and handles MT5/cTrader execution
- `risk_manager.py`
  - The5ers-hard risk rules
  - Absolute drawdown, daily loss, rollover window, max open trades
  - ATR-based lot sizing and SL/TP calculation
- `news_filter.py`
  - News window blocking logic

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
