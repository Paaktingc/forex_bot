"""
backtest_report.py

Simulates trades on historical data using the trained model.
Enforces SL/TP exits with no look-ahead bias and reports key
performance metrics.

TARGET: return >8 %, max drawdown <4 %.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Tuple

import config
from model import load_model, predict_signal
from features import FEATURE_COLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _pip_value(symbol: str = "EURUSD") -> float:
    """Returns the point size for a symbol (default 0.00001 for 5-digit pairs)."""
    return 0.00001


def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 24) -> float:
    """Annualised Sharpe ratio (risk-free = 0)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss.  Returns inf when no losses."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ─────────────────────────────────────────────────────────────
# core back-test engine
# ─────────────────────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    model,
    le,
    starting_balance: float = 10_000.0,
) -> dict:
    """
    Walk-forward backtest — no look-ahead.

    For each bar *i* the model predicts on the features at bar *i*.
    If a valid signal fires (confidence >= MIN_CONFIDENCE):
        • Entry  = close[i]
        • SL     = entry ∓ SL_ATR_MULT  * atr_14[i]
        • TP     = entry ± TP_ATR_MULT  * atr_14[i]
    The trade is held until SL, TP, or end of data is reached.

    Args:
        df: Full DataFrame that contains FEATURE_COLS + 'close' + 'atr_14' + 'high' + 'low'.
        model: Trained XGBClassifier.
        le: Fitted LabelEncoder.
        starting_balance: Notional starting equity for PnL%.

    Returns:
        dict of performance metrics.
    """
    required = {"close", "atr_14", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values

    balance = starting_balance
    peak_balance = balance
    max_drawdown_pct = 0.0
    trade_returns: list[float] = []

    i = 0
    n = len(df)

    while i < n - 1:
        # ── predict on bar i ──
        X_live = df[feature_cols].iloc[[i]]
        signal, confidence = predict_signal(model, le, X_live)

        if signal == 0 or np.isnan(atrs[i]) or atrs[i] <= 0:
            i += 1
            continue

        entry = closes[i]
        atr = atrs[i]

        if signal == 1:   # BUY
            sl = entry - config.SL_ATR_MULT * atr
            tp = entry + config.TP_ATR_MULT * atr
        elif signal == -1:  # SELL
            sl = entry + config.SL_ATR_MULT * atr
            tp = entry - config.TP_ATR_MULT * atr
        else:
            i += 1
            continue

        # ── walk forward to find exit (no look-ahead) ──
        exit_price = None
        for j in range(i + 1, n):
            if signal == 1:
                if lows[j] <= sl:
                    exit_price = sl
                    break
                if highs[j] >= tp:
                    exit_price = tp
                    break
            else:  # signal == -1
                if highs[j] >= sl:
                    exit_price = sl
                    break
                if lows[j] <= tp:
                    exit_price = tp
                    break

        if exit_price is None:
            # Still open at end of data — close at last close
            exit_price = closes[-1]
            j = n - 1

        # PnL in price units
        pnl = (exit_price - entry) if signal == 1 else (entry - exit_price)

        # Risk-based position sizing: risk RISK_PER_TRADE_PCT of balance
        risk_amount = balance * config.RISK_PER_TRADE_PCT
        risk_distance = config.SL_ATR_MULT * atr
        if risk_distance > 0:
            units = risk_amount / risk_distance
        else:
            units = 0

        trade_pnl = pnl * units
        balance += trade_pnl
        trade_returns.append(trade_pnl)

        # Drawdown tracking
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        max_drawdown_pct = max(max_drawdown_pct, dd)

        # Jump past the trade
        i = j + 1

    # ── aggregate metrics ──
    trade_returns_series = pd.Series(trade_returns) if trade_returns else pd.Series(dtype=float)
    total_trades = len(trade_returns)
    wins = int((trade_returns_series > 0).sum()) if total_trades > 0 else 0
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_pnl_pct = (balance - starting_balance) / starting_balance * 100
    sharpe = _sharpe_ratio(trade_returns_series)
    pf = _profit_factor(trade_returns_series)

    results = {
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": max_drawdown_pct * 100,
        "sharpe_ratio": sharpe,
        "profit_factor": pf,
        "final_balance": balance,
    }

    return results


# ─────────────────────────────────────────────────────────────
# report printer
# ─────────────────────────────────────────────────────────────
def print_report(results: dict) -> None:
    """Pretty-prints backtest results and flags target compliance."""
    print("\n" + "=" * 55)
    print("           BACKTEST PERFORMANCE REPORT")
    print("=" * 55)
    print(f"  Total Trades     : {results['total_trades']}")
    print(f"  Wins / Losses    : {results['wins']} / {results['total_trades'] - results['wins']}")
    print(f"  Win Rate         : {results['win_rate']:.2%}")
    print(f"  Total PnL %      : {results['total_pnl_pct']:+.2f}%")
    print(f"  Max Drawdown %   : {results['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio     : {results['sharpe_ratio']:.2f}")
    print(f"  Profit Factor    : {results['profit_factor']:.2f}")
    print(f"  Final Balance    : {results['final_balance']:,.2f}")
    print("-" * 55)

    # Target checks
    pnl_ok = results["total_pnl_pct"] > 8.0
    dd_ok = results["max_drawdown_pct"] < 4.0

    print(f"  Return >8%  : {'✅ PASS' if pnl_ok else '❌ FAIL'}")
    print(f"  Drawdown <4%: {'✅ PASS' if dd_ok else '❌ FAIL'}")
    print("=" * 55 + "\n")

    if not pnl_ok:
        logger.warning("Backtest return is below the 8% target.")
    if not dd_ok:
        logger.warning("Backtest max drawdown exceeds the 4% target.")


# ─────────────────────────────────────────────────────────────
# main entry-point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """
    Loads the trained model, reads historical feature + price data,
    runs the backtest simulation, and prints the report.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load model & encoder
    model, le = load_model()

    # Load full feature matrix (produced during labelling pipeline)
    x_path = config.DATA_DIR / "X_train.csv"
    if not os.path.exists(x_path):
        logger.error(f"Feature data not found at {x_path}. Run the labelling pipeline first.")
        return

    X_full = pd.read_csv(x_path)

    # We also need price columns for SL/TP simulation.
    # Look for the full labelled dataset that still has OHLCV + atr_14.
    full_path = config.DATA_DIR / "full_labelled.csv"
    if os.path.exists(full_path):
        df_full = pd.read_csv(full_path, index_col=0, parse_dates=True)
    else:
        # Fall back: try to reconstruct from X_train + raw data
        logger.warning(
            f"{full_path} not found. Attempting to load raw M15 CSV from data/ instead."
        )
        raw_path = config.DATA_DIR / "EURUSD_M15.csv"
        if not os.path.exists(raw_path):
            logger.error("No suitable price data found for backtesting.")
            return
        df_full = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # Align lengths (X_train may be shorter than df_full after labelling)
    min_len = min(len(X_full), len(df_full))
    df_full = df_full.iloc[:min_len].copy()
    X_full = X_full.iloc[:min_len].copy()

    # Merge feature columns into the price dataframe
    for col in X_full.columns:
        df_full[col] = X_full[col].values

    # Ensure required columns exist
    for req in ["close", "atr_14", "high", "low"]:
        if req not in df_full.columns:
            logger.error(f"Column '{req}' missing from backtest data.")
            return

    results = run_backtest(df_full, model, le)
    print_report(results)


if __name__ == "__main__":
    main()
