"""
backtest.py

Historical candle-by-candle backtesting for the THE5ERS forex bot.
Implements strict no-lookahead feature generation, trade simulation,
summary metrics, and walk-forward validation.
"""

from __future__ import annotations

import logging
import os
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
from data_feed import get_ohlcv_from_csv
from model import load_model, predict_signal, train_model
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

ENTRY_SPREAD = 0.00012
PIP_SIZE = 0.0001
PIP_VALUE_PER_LOT = 10.0
ANNUALIZATION_FACTOR = sqrt(252 * 24 * 4)

PASS_CRITERIA = {
    "total_return": ("Total return", 8.0, lambda value: 8.0 <= value <= 40.0, "8-40%"),
    "max_drawdown": ("Max drawdown", 4.0, lambda value: value < 4.0, "< 4%"),
    "win_rate": ("Win rate", 45.0, lambda value: value > 45.0, "> 45%"),
    "profit_factor": ("Profit factor", 1.3, lambda value: value > 1.3, "> 1.3"),
    "sharpe_ratio": ("Sharpe ratio", 0.8, lambda value: value > 0.8, "> 0.8"),
    "min_trades": ("Min trades", 50, lambda value: value > 50, "> 50"),
}

ENABLE_WALK_FORWARD = os.getenv("BACKTEST_FULL_VALIDATION", "0") == "1"
ENABLE_MONTE_CARLO = os.getenv("BACKTEST_MONTE_CARLO", "0") == "1"
DEFAULT_MAX_M15_ROWS = int(os.getenv("BACKTEST_MAX_M15_ROWS", "3000"))


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        index = out.index
    elif "time" in out.columns:
        index = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out = out.drop(columns=["time"])
    else:
        index = pd.to_datetime(out.index, utc=True, errors="coerce")

    if getattr(index, "tz", None) is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    out.index = index
    out = out[~out.index.isna()]
    out = out.sort_index()
    return out


def build_feature_matrix(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    from features import build_feature_matrix as _build_feature_matrix

    return _build_feature_matrix(df_m15, df_h1)


def apply_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
    from labelling import apply_triple_barrier as _apply_triple_barrier

    return _apply_triple_barrier(df)


def prepare_training_data(df: pd.DataFrame):
    from labelling import prepare_training_data as _prepare_training_data

    return _prepare_training_data(df)


def _news_events_path() -> Path:
    return config.DATA_DIR / "news_events.csv"


def _load_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price data not found at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Price data at {path} is empty")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.set_index("time")
    else:
        first_col = df.columns[0]
        parsed_index = pd.to_datetime(df[first_col], utc=True, errors="coerce")
        if parsed_index.notna().all():
            df = df.drop(columns=[first_col])
            df.index = parsed_index
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[~df.index.isna()].sort_index()
    return df


def _empty_metrics() -> dict[str, Any]:
    metrics = {
        "total_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "profit_factor": 0.0,
        "win_rate_pct": 0.0,
        "total_trades": 0,
        "avg_trade_duration_candles": 0.0,
    }
    metrics.update(_evaluate_pass_criteria(metrics))
    return metrics


def _evaluate_pass_criteria(metrics: dict[str, Any]) -> dict[str, Any]:
    values = {
        "total_return": float(metrics.get("total_return_pct", 0.0) or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
        "win_rate": float(metrics.get("win_rate_pct", 0.0) or 0.0),
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0) or 0.0),
        "min_trades": int(metrics.get("total_trades", 0) or 0),
    }

    pass_criteria = {
        name: rule(values[name])
        for name, (_, _, rule, _) in PASS_CRITERIA.items()
    }
    return {
        "pass_criteria": pass_criteria,
        "overall_pass": all(pass_criteria.values()),
    }


def _format_value(name: str, metrics: dict[str, Any]) -> str:
    if name == "min_trades":
        return str(int(metrics.get("total_trades", 0) or 0))
    key_map = {
        "total_return": "total_return_pct",
        "max_drawdown": "max_drawdown_pct",
        "win_rate": "win_rate_pct",
        "profit_factor": "profit_factor",
        "sharpe_ratio": "sharpe_ratio",
    }
    value = float(metrics.get(key_map[name], 0.0) or 0.0)
    if name in {"total_return", "max_drawdown", "win_rate"}:
        return f"{value:.2f}%"
    return f"{value:.2f}"


def _box_line(text: str) -> str:
    return f"│ {text:<39}│"


def _print_metric_block(
    metrics: dict[str, Any],
    *,
    date_range: tuple[pd.Timestamp, pd.Timestamp],
    total_bars: int,
) -> None:
    start, end = date_range
    print("┌─────────────────────────────────────────┐")
    print(_box_line("BACKTEST RESULTS"))
    print("├─────────────────────────────────────────┤")
    print(_box_line(f"Date Range:    {start}"))
    print(_box_line(f"               {end}"))
    print(_box_line(f"Total Bars:    {total_bars}"))
    print(_box_line(f"Total Trades:  {metrics['total_trades']}"))
    print(_box_line(""))
    print(_box_line(f"Total Return:  {metrics['total_return_pct']:.2f}%   TARGET: 8-40%"))
    print(_box_line(f"Max Drawdown:  {metrics['max_drawdown_pct']:.2f}%   TARGET: <4%"))
    print(_box_line(f"Win Rate:      {metrics['win_rate_pct']:.2f}%   TARGET: >45%"))
    print(_box_line(f"Profit Factor: {metrics['profit_factor']:.2f}    TARGET: >1.3"))
    print(_box_line(f"Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}    TARGET: >0.8"))
    print(_box_line(f"Total Trades:  {metrics['total_trades']}       TARGET: >50"))
    print(_box_line(""))
    print(_box_line("PASS/FAIL per metric"))
    for key, (label, _, _, _) in PASS_CRITERIA.items():
        status = "PASS" if metrics.get("pass_criteria", {}).get(key, False) else "FAIL"
        print(_box_line(f"{label}: {status}"))
    print("└─────────────────────────────────────────┘")


def _report_overall_status(metrics: dict[str, Any]) -> None:
    failing = [name for name, passed in metrics.get("pass_criteria", {}).items() if not passed]
    if not failing:
        print("\n✅ READY FOR DEMO")
        return

    fixes = {
        "total_return": "Reduce overfitting or improve signal realism before using this result.",
        "max_drawdown": "Tighten risk sizing or exits to bring drawdown under control.",
        "win_rate": "Retune the classifier and signal threshold to improve precision.",
        "profit_factor": "Improve trade quality before proceeding to demo conditions.",
        "sharpe_ratio": "Stabilize returns; inspect trade clustering and sizing.",
        "min_trades": "Lower confidence or rebalance labels to increase signal frequency.",
    }
    print("\n❌ NOT READY")
    for metric in failing:
        print(f"- {PASS_CRITERIA[metric][0]}: {fixes[metric]}")


class BacktestEngine:
    def __init__(self, df_m15, df_h1, starting_balance=10000):
        self.df = _to_utc_index(df_m15)
        self.df_h1 = _to_utc_index(df_h1)
        self.balance = float(starting_balance)
        self.starting_balance = float(starting_balance)
        self.equity_curve: list[float] = []
        self.trades: list[dict[str, Any]] = []

        self._validate_inputs()
        self.risk_manager = RiskManager(self.starting_balance)
        self.news_events = self._load_news_events()
        self.feature_frame = build_feature_matrix(self.df, self.df_h1)
        self.model = None
        self.label_encoder = None
        try:
            self.model, self.label_encoder = load_model()
        except Exception as exc:
            logger.warning(
                "BacktestEngine could not load saved model artifacts at init: %s",
                exc,
            )

    def _validate_inputs(self) -> None:
        missing_m15 = {"open", "high", "low", "close"} - set(self.df.columns)
        missing_h1 = {"close"} - set(self.df_h1.columns)
        if missing_m15:
            raise ValueError(f"df_m15 missing required columns: {sorted(missing_m15)}")
        if missing_h1:
            raise ValueError(f"df_h1 missing required columns: {sorted(missing_h1)}")

    def _load_news_events(self) -> pd.DataFrame:
        path = _news_events_path()
        if not path.exists():
            logger.warning(
                "Historical news file not found at %s. News filter disabled for backtests.",
                path,
            )
            return pd.DataFrame(columns=["datetime_utc", "currency"])

        df = pd.read_csv(path)
        required = {"datetime_utc", "currency"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "Historical news file missing columns %s. News filter disabled for backtests.",
                sorted(missing),
            )
            return pd.DataFrame(columns=["datetime_utc", "currency"])

        df = df.copy()
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        df["currency"] = df["currency"].astype(str).str.upper()
        df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
        return df

    def _historical_rollover_window(self, timestamp: pd.Timestamp) -> bool:
        hour = timestamp.tz_convert("UTC").hour
        return config.ROLLOVER_START_UTC <= hour < config.ROLLOVER_END_UTC

    def _historical_news_window(self, timestamp: pd.Timestamp) -> bool:
        if self.news_events.empty:
            return False

        currencies = {config.SYMBOL[:3].upper(), config.SYMBOL[3:6].upper()}
        relevant = self.news_events[self.news_events["currency"].isin(currencies)]
        if relevant.empty:
            return False

        delta = relevant["datetime_utc"] - timestamp
        minutes = delta.abs().dt.total_seconds() / 60.0
        return bool((minutes <= config.NEWS_BUFFER_MINS).any())

    def _ensure_model_ready(self) -> None:
        if self.model is None or self.label_encoder is None:
            raise ValueError(
                "BacktestEngine model artifacts are not loaded. "
                "Load saved artifacts or inject a trained model and encoder before run()."
            )

    def _build_feature_row(self, bar_idx: int) -> pd.Series | None:
        if bar_idx <= 0:
            return None
        signal_time = self.df.index[bar_idx - 1]
        if self.feature_frame.empty or signal_time not in self.feature_frame.index:
            return None
        return self.feature_frame.loc[signal_time]

    def _trade_pnl_currency(self, signal: int, entry_price: float, exit_price: float, lot: float) -> float:
        price_move = exit_price - entry_price
        if signal == -1:
            price_move = -price_move
        pnl_pips = price_move / PIP_SIZE
        return round(pnl_pips * PIP_VALUE_PER_LOT * lot, 2)

    def _trade_result_from_pnl(self, pnl_currency: float) -> str:
        if pnl_currency > 0:
            return "WIN"
        if pnl_currency < 0:
            return "LOSS"
        return "BE"

    def _open_trade(self, bar_idx: int, signal: int, confidence: float, atr: float) -> dict[str, Any] | None:
        raw_open = float(self.df["open"].iloc[bar_idx])
        entry_price = raw_open + ENTRY_SPREAD if signal == 1 else raw_open - ENTRY_SPREAD
        entry_price = round(float(entry_price), 5)
        sl, tp = self.risk_manager.calculate_sl_tp(signal, entry_price, float(atr))
        lot = self.risk_manager.calculate_lot_size(self.balance, sl, entry_price, config.SYMBOL)
        if lot <= 0:
            return None

        return {
            "signal": signal,
            "entry_idx": bar_idx,
            "entry_time": self.df.index[bar_idx],
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "confidence": float(confidence),
            "bars_open": 0,
            "result": None,
        }

    def _close_trade(
        self,
        trade: dict[str, Any],
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
        forced_result: str | None = None,
    ) -> None:
        duration = int(trade["bars_open"])
        pnl_currency = self._trade_pnl_currency(
            signal=int(trade["signal"]),
            entry_price=float(trade["entry_price"]),
            exit_price=float(exit_price),
            lot=float(trade["lot"]),
        )
        result = forced_result or self._trade_result_from_pnl(pnl_currency)
        self.balance = round(self.balance + pnl_currency, 2)
        self.trades.append(
            {
                **trade,
                "exit_price": round(float(exit_price), 5),
                "exit_time": exit_time,
                "exit_reason": exit_reason,
                "duration_candles": duration,
                "pnl_currency": pnl_currency,
                "result": result,
            }
        )

    def _update_open_trades(self, open_trades: list[dict[str, Any]], bar_idx: int) -> list[dict[str, Any]]:
        high = float(self.df["high"].iloc[bar_idx])
        low = float(self.df["low"].iloc[bar_idx])
        close = float(self.df["close"].iloc[bar_idx])
        timestamp = self.df.index[bar_idx]
        remaining: list[dict[str, Any]] = []

        for trade in open_trades:
            if bar_idx <= int(trade["entry_idx"]):
                remaining.append(trade)
                continue

            trade["bars_open"] = int(bar_idx - int(trade["entry_idx"]))
            signal = int(trade["signal"])
            sl = float(trade["sl"])
            tp = float(trade["tp"])

            if signal == 1:
                sl_hit = low <= sl
                tp_hit = high >= tp
            else:
                sl_hit = high >= sl
                tp_hit = low <= tp

            if sl_hit:
                self._close_trade(trade, sl, timestamp, "SL", forced_result="LOSS")
                continue
            if tp_hit:
                self._close_trade(trade, tp, timestamp, "TP", forced_result="WIN")
                continue
            if int(trade["bars_open"]) >= config.TRIPLE_BARRIER_TIME_LIMIT:
                self._close_trade(trade, close, timestamp, "TIME_LIMIT", forced_result="BE")
                continue

            remaining.append(trade)

        return remaining

    def _close_end_of_data_trades(self, open_trades: list[dict[str, Any]]) -> None:
        if not open_trades:
            return

        final_timestamp = self.df.index[-1]
        final_close = float(self.df["close"].iloc[-1])
        for trade in open_trades:
            trade["bars_open"] = max(int(len(self.df) - 1 - int(trade["entry_idx"])), 0)
            self._close_trade(trade, final_close, final_timestamp, "END_OF_DATA")

    def run(self) -> dict:
        self._ensure_model_ready()
        self.balance = self.starting_balance
        self.equity_curve = []
        self.trades = []
        self.risk_manager = RiskManager(self.starting_balance)

        open_trades: list[dict[str, Any]] = []

        for bar_idx in range(101, len(self.df)):
            feature_row = self._build_feature_row(bar_idx)
            current_time = self.df.index[bar_idx]

            if (
                feature_row is not None
                and not self._historical_rollover_window(current_time)
                and not self._historical_news_window(current_time)
                and len(open_trades) < config.MAX_OPEN_TRADES
            ):
                X_live = feature_row.to_frame().T
                signal, confidence = predict_signal(self.model, self.label_encoder, X_live)
                atr_value = float(feature_row.get("atr_14", 0.0) or 0.0)

                if signal != 0 and atr_value > 0:
                    trade = self._open_trade(bar_idx, signal, confidence, atr_value)
                    if trade is not None:
                        open_trades.append(trade)

            open_trades = self._update_open_trades(open_trades, bar_idx)
            self.equity_curve.append(float(self.balance))

        self._close_end_of_data_trades(open_trades)
        if self.equity_curve:
            self.equity_curve[-1] = float(self.balance)
        else:
            self.equity_curve.append(float(self.balance))

        metrics = self.compute_metrics(self.equity_curve)
        metrics["ending_balance"] = round(self.balance, 2)
        metrics.update(_evaluate_pass_criteria(metrics))
        return metrics

    def compute_metrics(self, equity_curve: list) -> dict:
        if not equity_curve:
            return _empty_metrics()

        curve = np.array([self.starting_balance] + [float(x) for x in equity_curve], dtype=float)
        total_return_pct = (curve[-1] - self.starting_balance) / self.starting_balance * 100.0

        peaks = np.maximum.accumulate(curve)
        drawdowns = np.where(peaks > 0, (peaks - curve) / peaks, 0.0)
        max_drawdown_pct = float(drawdowns.max() * 100.0)

        returns = pd.Series(curve).pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float(returns.mean() / returns.std() * ANNUALIZATION_FACTOR)

        pnls = pd.Series([float(trade["pnl_currency"]) for trade in self.trades], dtype=float)
        gross_profit = float(pnls[pnls > 0].sum()) if not pnls.empty else 0.0
        gross_loss = float(abs(pnls[pnls < 0].sum())) if not pnls.empty else 0.0
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        total_trades = len(self.trades)
        wins = sum(1 for trade in self.trades if trade["result"] == "WIN")
        win_rate_pct = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_duration = (
            float(np.mean([trade["duration_candles"] for trade in self.trades]))
            if self.trades
            else 0.0
        )

        metrics = {
            "total_return_pct": round(float(total_return_pct), 2),
            "max_drawdown_pct": round(float(max_drawdown_pct), 2),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "profit_factor": float(profit_factor) if np.isinf(profit_factor) else round(float(profit_factor), 4),
            "win_rate_pct": round(float(win_rate_pct), 2),
            "total_trades": int(total_trades),
            "avg_trade_duration_candles": round(float(avg_duration), 2),
        }
        metrics.update(_evaluate_pass_criteria(metrics))
        return metrics

    def _prepare_training_frame(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
        feature_frame = build_feature_matrix(df_m15, df_h1)
        if feature_frame.empty:
            return pd.DataFrame()

        raw_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in df_m15.columns]
        train_frame = df_m15.loc[feature_frame.index, raw_cols].copy()
        for column in feature_frame.columns:
            train_frame[column] = feature_frame[column]
        return train_frame

    def walk_forward_validation(self) -> list[dict]:
        results: list[dict[str, Any]] = []
        splitter = TimeSeriesSplit(n_splits=5)

        for window, (train_idx, raw_test_idx) in enumerate(splitter.split(self.df), start=1):
            test_idx = raw_test_idx[1:]
            if len(test_idx) == 0:
                continue
            train_m15 = self.df.iloc[train_idx].copy()
            test_m15 = self.df.iloc[test_idx].copy()

            train_end = train_m15.index[-1]
            test_end = test_m15.index[-1]
            train_h1 = self.df_h1.loc[self.df_h1.index <= train_end].copy()
            test_h1 = self.df_h1.loc[
                (self.df_h1.index >= test_m15.index[0]) & (self.df_h1.index <= test_end)
            ].copy()

            fold_metrics = _empty_metrics()
            fold_metrics["window"] = window
            fold_metrics["train_start"] = str(train_m15.index[0])
            fold_metrics["train_end"] = str(train_end)
            fold_metrics["test_start"] = str(test_m15.index[0])
            fold_metrics["test_end"] = str(test_end)
            fold_metrics["gap_bars"] = int(test_idx[0] - train_idx[-1] - 1)

            try:
                training_frame = self._prepare_training_frame(train_m15, train_h1)
                if training_frame.empty:
                    raise ValueError("training features are empty")

                labelled = apply_triple_barrier(training_frame)
                if labelled.empty or "label" not in labelled.columns:
                    raise ValueError("label generation failed")

                X_train, y_encoded, le = prepare_training_data(labelled)
                if X_train.empty or len(np.unique(y_encoded)) < 2:
                    raise ValueError("insufficient class diversity for training")

                fold_model = train_model(
                    X_train,
                    y_encoded,
                    label_encoder=le,
                    persist=False,
                    verbose_reports=False,
                )
                fold_engine = BacktestEngine(test_m15, test_h1, starting_balance=self.starting_balance)
                fold_engine.model = fold_model
                fold_engine.label_encoder = le
                fold_metrics = fold_engine.run()
                fold_metrics["window"] = window
            except Exception as exc:
                logger.warning("Walk-forward window %s failed: %s", window, exc)
                fold_metrics["error"] = str(exc)

            results.append(fold_metrics)

        average_metrics = {"window": "average"}
        numeric_keys = [
            "total_return_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "profit_factor",
            "win_rate_pct",
            "total_trades",
            "avg_trade_duration_candles",
        ]
        for key in numeric_keys:
            values = [float(result[key]) for result in results if key in result]
            average_metrics[key] = round(float(np.mean(values)), 4) if values else 0.0
        average_metrics.update(_evaluate_pass_criteria(average_metrics))
        results.append(average_metrics)
        return results

    def run_monte_carlo(self, n: int = 1000, seed: int | None = 42) -> dict:
        """
        Bootstrap Monte Carlo on the realized trade P&L sequence.

        Resamples the per-trade P&L with replacement (same number of trades)
        `n` times, rebuilds an equity curve for each draw, and records the
        peak-to-trough max drawdown per iteration. This stress-tests how trade
        order / sampling luck affects drawdown under the current sizing.

        run() must be called first to populate self.trades.

        NOTE: each resampled P&L embeds the lot size from the original backtest
        path; the dynamic drawdown circuit breaker is NOT re-applied per draw,
        so this assumes fixed per-trade position sizing.

        Returns a dict: pct_under_5pct, dd_95th_pct, worst_dd_pct, mean_dd_pct,
        n_iterations, n_trades.
        """
        pnls = np.array(
            [float(trade["pnl_currency"]) for trade in self.trades], dtype=float
        )
        if pnls.size == 0:
            logger.warning("run_monte_carlo: no trades to resample.")
            return {
                "pct_under_5pct": 0.0,
                "dd_95th_pct": 0.0,
                "worst_dd_pct": 0.0,
                "mean_dd_pct": 0.0,
                "n_iterations": 0,
                "n_trades": 0,
            }

        rng = np.random.default_rng(seed)
        m = pnls.size

        # Resample with replacement → shape (n, m).
        draws = rng.choice(pnls, size=(n, m), replace=True)

        # Equity curve per iteration, anchored at starting_balance.
        equity = self.starting_balance + np.cumsum(draws, axis=1)
        equity = np.column_stack(
            [np.full(n, self.starting_balance, dtype=float), equity]
        )

        peaks = np.maximum.accumulate(equity, axis=1)
        drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
        max_dd_pct = drawdowns.max(axis=1) * 100.0

        return {
            "pct_under_5pct": float(np.mean(max_dd_pct < 5.0) * 100.0),
            "dd_95th_pct": float(np.percentile(max_dd_pct, 95)),
            "worst_dd_pct": float(max_dd_pct.max()),
            "mean_dd_pct": float(max_dd_pct.mean()),
            "n_iterations": int(n),
            "n_trades": int(m),
        }


def _print_monte_carlo_block(result: dict) -> None:
    print("\nMONTE CARLO (1,000× bootstrap on trade P&L)")
    print("-" * 56)
    print(f"Iterations:           {result['n_iterations']}")
    print(f"Trades per iteration: {result['n_trades']}")
    print(f"Runs with DD < 5%:    {result['pct_under_5pct']:.1f}%   TARGET: >95%")
    print(f"95th percentile DD:   {result['dd_95th_pct']:.2f}%")
    print(f"Worst-case DD:        {result['worst_dd_pct']:.2f}%")
    print(f"Mean DD:              {result['mean_dd_pct']:.2f}%")
    status = "PASS" if result["pct_under_5pct"] > 95.0 else "FAIL"
    print(f"Monte Carlo target:   {status} (>95% of runs under 5%)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("risk_manager").setLevel(logging.ERROR)

    try:
        df_m15 = get_ohlcv_from_csv("EURUSD", "M15")
        df_h1 = get_ohlcv_from_csv("EURUSD", "H1")
        if DEFAULT_MAX_M15_ROWS > 0 and len(df_m15) > DEFAULT_MAX_M15_ROWS:
            df_m15 = df_m15.iloc[-DEFAULT_MAX_M15_ROWS:].copy()
            first_timestamp = df_m15.index[0].floor("h")
            df_h1 = df_h1.loc[df_h1.index >= first_timestamp].copy()
    except Exception as exc:
        logger.error("Failed to load historical CSVs: %s", exc)
        return

    try:
        engine = BacktestEngine(df_m15, df_h1)
        if engine.model is None or engine.label_encoder is None:
            logger.info("Training a local model for backtest execution.")
            training_frame = engine._prepare_training_frame(df_m15, df_h1)
            labelled = apply_triple_barrier(training_frame)
            X_train, y_encoded, le = prepare_training_data(labelled)
            engine.model = train_model(
                X_train,
                y_encoded,
                label_encoder=le,
                persist=False,
                verbose_reports=False,
            )
            engine.label_encoder = le
        metrics = engine.run()
        walk_forward_results = engine.walk_forward_validation() if ENABLE_WALK_FORWARD else []
    except Exception as exc:
        logger.error("Backtest execution failed: %s", exc)
        return

    _print_metric_block(
        metrics,
        date_range=(df_m15.index[0], df_m15.index[-1]),
        total_bars=len(df_m15),
    )

    if metrics["total_return_pct"] > 200.0:
        print("\n⛔ STOPPED: Return exceeds 200% — likely lookahead bias")
        print("Check: features built without future data, no label leakage")
        sys.exit(1)

    if ENABLE_WALK_FORWARD:
        print("\nWALK-FORWARD VALIDATION")
        print("-" * 56)
        for result in walk_forward_results[:-1]:
            print(
                f"Window {result['window']}: "
                f"return={result['total_return_pct']:.2f}% "
                f"dd={result['max_drawdown_pct']:.2f}% "
                f"wr={result['win_rate_pct']:.2f}% "
                f"pf={result['profit_factor']:.2f} "
                f"sharpe={result['sharpe_ratio']:.2f} "
                f"trades={int(result['total_trades'])}"
            )

        average = walk_forward_results[-1]
        print(
            f"Average : return={average['total_return_pct']:.2f}% "
            f"dd={average['max_drawdown_pct']:.2f}% "
            f"wr={average['win_rate_pct']:.2f}% "
            f"pf={average['profit_factor']:.2f} "
            f"sharpe={average['sharpe_ratio']:.2f} "
            f"trades={int(average['total_trades'])}"
        )
    else:
        print("\nWalk-forward validation skipped. Set BACKTEST_FULL_VALIDATION=1 to enable it.")

    if ENABLE_MONTE_CARLO:
        _print_monte_carlo_block(engine.run_monte_carlo(n=1000))

    _report_overall_status(metrics)


if __name__ == "__main__":
    main()
