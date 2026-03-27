# THE5ERS FOREX ML BOT

A fully automated ML forex trading bot connecting Python to MetaTrader 5 (MT5) designed to pass The5ers Bootcamp prop firm evaluation.

## Requirements
- Python 3.11+
- Windows OS (MetaTrader5 library requirement)
- MetaTrader 5 Terminal installed and running

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
   Copy the `.env.example` file to `.env` and fill in your MetaTrader 5 credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to include your `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER`.

4. **Verify Configurations**:
   Check `config.py` to reflect your specific phase thresholds, risk constraints, and parameters.

## Running the Bot

- **Testing**: Run `pytest tests/` to execute unit tests locally (a MacOS adapter/mock may be required if MT5 is absent).
- **Main Bot**: Run `python main.py` to start the live trading loop natively on a Windows endpoint. Ensure MT5 is open or AutoLogin via the provided `.env`.

This bot enforces strict daily limits (4%) and absolute drawdowns (4.5%), incorporates Machine Learning inference, and safeguards against rollover (21:00-22:00 UTC) and high-impact news windows.
