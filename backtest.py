"""
backtest.py

Historical candle-by-candle backtesting for the The5ers Bootcamp bot.

Primary engine: RulesBacktestEngine — backtests the RULES strategy
(strategy.py, not the ML model) with conservative costs and the live
pacing/risk gates, then feeds a Bootcamp step simulator + Monte Carlo.

Single GO/NO-GO command:
    python backtest.py --go-no-go
GO only if P(breach −5%) < 1%, P(kill switch) < 10%, P(pass) > 70%, and
PF ≥ 1.25 after costs in every walk-forward fold.

The legacy model-driven BacktestEngine remains for the research pipeline.
"""

from __future__ import annotations

import argparse
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
import strategy as strategy_module
from data_feed import get_ohlcv_from_csv
from model import load_model, predict_signal, train_model
from risk_manager import RiskManager, compute_sl_tp
from symbol_specs import get_symbol_spec

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
        sl_tp = self.risk_manager.calculate_sl_tp(signal, entry_price, float(atr))
        if sl_tp is None:
            return None
        sl, tp = sl_tp
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


# ===========================================================================
# Rules-strategy backtest (The5ers Bootcamp)
# ===========================================================================

GO_CRITERIA = {
    "p_breach_official": ("P(breach −5%)", lambda v: v < 0.01, "< 1%"),
    "p_kill_switch": ("P(hit −3% kill switch)", lambda v: v < 0.10, "< 10%"),
    "p_pass": ("P(+6% before −5%)", lambda v: v > 0.70, "> 70%"),
    "fold_pf": ("PF ≥ 1.25 after costs in every fold", lambda v: v, "all folds"),
}


class RulesBacktestEngine:
    """
    Bar-by-bar simulation of the rules strategy with:
      - entry at next candle open, spread floored at
        BACKTEST_SPREAD_FLOOR_PIPS + entry slippage
      - SL exits filled with stop-out slippage (news slippage inside
        historical news windows when data/news_events.csv exists)
      - commission BACKTEST_COMMISSION_PER_LOT_RT per lot round trip
      - breakeven SL move at +1R (from the NEXT bar, conservative)
      - live pacing gates: sessions, rollover/server window, 2 trades/day,
        2 consecutive losses/day, 5/week, −0.75% day stop, −1.5% week stop
      - kill-switch tracking at −3% from starting balance (reported; set
        stop_on_kill=True to also halt the simulation there)
    """

    _BE_CONFIG = "config"  # sentinel: be_at_r follows config.BE_AT_R

    def __init__(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        params: strategy_module.StrategyParams | None = None,
        starting_balance: float = 10_000.0,
        stop_on_kill: bool = False,
        sl_atr_mult: float | None = None,
        tp_r: float | None = None,
        be_at_r: float | None | str = _BE_CONFIG,
        collect_diagnostics: bool = False,
        symbol: str | None = None,
    ) -> None:
        self.df = _to_utc_index(df_m15)
        self.df_h1 = _to_utc_index(df_h1)
        self.params = params or strategy_module.StrategyParams()
        self.starting_balance = float(starting_balance)
        self.stop_on_kill = stop_on_kill
        self.symbol = (symbol or config.SYMBOL).upper()
        # Exit-parameter overrides exist for research sweeps only;
        # be_at_r=None disables the breakeven move entirely.
        self.sl_atr_mult = config.SL_ATR_MULT if sl_atr_mult is None else sl_atr_mult
        self.tp_r = config.TP_R if tp_r is None else tp_r
        self.be_at_r = config.BE_AT_R if be_at_r == self._BE_CONFIG else be_at_r
        self.collect_diagnostics = collect_diagnostics
        self.funnel: dict[str, int] = {}
        self.shadow_trades: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.news_events = self._load_news_events()

        spec = get_symbol_spec(self.symbol)
        if spec is None:
            raise ValueError(f"No symbol spec for {self.symbol}")
        self.pip = spec.pip_size
        self.pip_value = spec.pip_value_per_standard_lot
        self.min_lot = spec.min_lot
        self.lot_step = spec.lot_step
        self.max_lot = spec.max_lot

        spread_floor = config.BACKTEST_SPREAD_FLOOR_BY_SYMBOL.get(
            self.symbol, config.BACKTEST_SPREAD_FLOOR_PIPS
        )
        self.spread = spread_floor * self.pip
        self.slip_entry = config.BACKTEST_SLIPPAGE_ENTRY_PIPS * self.pip
        self.slip_stop = config.BACKTEST_SLIPPAGE_STOP_PIPS * self.pip
        self.slip_news = config.BACKTEST_SLIPPAGE_NEWS_PIPS * self.pip

        # Non-FX transfer research (cycle 5): thresholds/slippage become
        # fractions of price (config.BP_THRESHOLDS); FX path is unchanged.
        self.bp_mode = spec.asset_class != "fx"
        self.asset_class = spec.asset_class
        self.contract_size = config.CONTRACT_SIZE_BY_SYMBOL.get(self.symbol, 100_000)

    # -- asset-class aware cost/threshold helpers (FX path unchanged) -------

    def _entry_cost(self, price: float) -> float:
        if self.bp_mode:
            return self.spread + price * config.BP_THRESHOLDS["slip_entry"]
        return self.spread + self.slip_entry

    def _stop_slip(self, price: float, near_news: bool) -> float:
        if self.bp_mode:
            key = "slip_news" if near_news else "slip_stop"
            return price * config.BP_THRESHOLDS[key]
        return self.slip_news if near_news else self.slip_stop

    def _commission_per_lot(self, price: float) -> float:
        pct = config.BACKTEST_COMMISSION_PCT_PER_SIDE_BY_SYMBOL.get(self.symbol)
        if pct is not None:  # metals: percentage commission (ASSUMED rate)
            return 2.0 * pct * price * self.contract_size
        if self.asset_class == "index":  # The5ers: indices commission-free
            return 0.0
        return config.BACKTEST_COMMISSION_PER_LOT_RT

    def _atr_floor_ok(self, atr: float, price: float) -> bool:
        if self.bp_mode:
            return atr >= price * config.BP_THRESHOLDS["atr_min"]
        return atr / self.pip >= config.ATR_MIN_PIPS

    def _sl_tp(self, sig: int, entry: float, atr: float, swing: float | None):
        """Clamped SL/TP: FX uses compute_sl_tp; bp-mode uses the same
        formula with the price-relative clamp."""
        if not self.bp_mode:
            return compute_sl_tp(
                sig, entry, atr, swing_price=swing, symbol=self.symbol,
                sl_atr_mult=self.sl_atr_mult, tp_r=self.tp_r,
            )
        anchor = swing if swing is not None else entry
        sl = anchor - sig * self.sl_atr_mult * atr
        sl_distance = (entry - sl) * sig
        if sl_distance <= 0:
            return None
        if not (
            entry * config.BP_THRESHOLDS["sl_min"]
            <= sl_distance
            <= entry * config.BP_THRESHOLDS["sl_max"]
        ):
            return None
        tp = entry + sig * self.tp_r * sl_distance
        return round(sl, 5), round(tp, 5)

    def _load_news_events(self) -> pd.DataFrame:
        path = _news_events_path()
        if not path.exists():
            logger.warning(
                "No historical news file at %s — backtest runs without news "
                "blackouts (live trading fails closed instead).",
                path,
            )
            return pd.DataFrame(columns=["datetime_utc", "currency"])
        df = pd.read_csv(path)
        if not {"datetime_utc", "currency"} <= set(df.columns):
            return pd.DataFrame(columns=["datetime_utc", "currency"])
        df = df.copy()
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        currencies = set(
            config.NEWS_CURRENCIES_BY_SYMBOL.get(
                self.symbol, (self.symbol[:3], self.symbol[3:6])
            )
        )
        df = df.dropna(subset=["datetime_utc"])
        df = df[df["currency"].astype(str).str.upper().isin(currencies)]
        return df.sort_values("datetime_utc")

    def _in_news_window(self, ts: pd.Timestamp, buffer_mins: float | None = None) -> bool:
        if self.news_events.empty:
            return False
        buffer_mins = buffer_mins or config.NEWS_BUFFER_MINS
        minutes = (self.news_events["datetime_utc"] - ts).abs().dt.total_seconds() / 60.0
        return bool((minutes <= buffer_mins).any())

    def _lot_size(self, balance: float, sl_distance: float) -> float:
        risk_amount = balance * config.RISK_PER_TRADE_PCT
        sl_pips = sl_distance / self.pip
        lot = risk_amount / (sl_pips * self.pip_value)
        lot = min(lot, self.max_lot)
        lot = np.floor(lot / self.lot_step + 1e-12) * self.lot_step
        lot = round(float(lot), 2)
        return lot if lot >= self.min_lot else 0.0

    def _news_mask(self, index: pd.DatetimeIndex, buffer_mins: float) -> np.ndarray:
        """Boolean per-bar mask: bar timestamp within ±buffer of any event."""
        mask = np.zeros(len(index), dtype=bool)
        if self.news_events.empty:
            return mask
        buffer = pd.Timedelta(minutes=buffer_mins)
        for ts in self.news_events["datetime_utc"]:
            mask |= (index >= ts - buffer) & (index <= ts + buffer)
        return mask

    def _walk_exit(
        self,
        i_entry: int,
        direction: int,
        entry: float,
        sl: float,
        tp: float,
        risk_distance: float,
        use_be: bool,
    ) -> dict[str, Any]:
        """
        Walks bars after i_entry with the SAME exit rules as the live loop
        (SL-first conservatism, news slippage on stops, BE move effective
        from the bar after the trigger). Zero account impact — used for
        shadow candidates and no-BE counterfactuals. Returns R after costs.
        """
        highs, lows, closes = self._highs, self._lows, self._closes
        d = direction
        cur_sl = sl
        be_moved = not use_be or self.be_at_r is None
        be_trigger = entry + d * (self.be_at_r or 0.0) * risk_distance
        hh, ll = -np.inf, np.inf
        commission_r = (
            self._commission_per_lot(entry)
            / ((risk_distance / self.pip) * self.pip_value)
        )

        n = len(highs)
        for j in range(i_entry + 1, n):
            hh = max(hh, highs[j])
            ll = min(ll, lows[j])
            sl_hit = lows[j] <= cur_sl if d == 1 else highs[j] >= cur_sl
            tp_hit = highs[j] >= tp if d == 1 else lows[j] <= tp
            if sl_hit:
                slip = self._stop_slip(entry, bool(self._news_bar_mask[j]))
                exit_price = cur_sl - d * slip
                reason = "SL"
            elif tp_hit:
                exit_price, reason = tp, "TP"
            else:
                if not be_moved:
                    reached = highs[j] >= be_trigger if d == 1 else lows[j] <= be_trigger
                    if reached:
                        cur_sl = entry
                        be_moved = True
                continue
            r = (exit_price - entry) * d / risk_distance - commission_r
            mfe = ((hh - entry) if d == 1 else (entry - ll)) / risk_distance
            mae = ((entry - ll) if d == 1 else (hh - entry)) / risk_distance
            return {
                "r": r, "exit_reason": reason, "exit_idx": j,
                "mfe_r": mfe, "mae_r": mae,
            }

        exit_price = closes[-1]
        r = (exit_price - entry) * d / risk_distance - commission_r
        mfe = ((hh - entry) if d == 1 else (entry - ll)) / risk_distance if n > i_entry + 1 else 0.0
        mae = ((entry - ll) if d == 1 else (hh - entry)) / risk_distance if n > i_entry + 1 else 0.0
        return {
            "r": r, "exit_reason": "END_OF_DATA", "exit_idx": n - 1,
            "mfe_r": mfe, "mae_r": mae,
        }

    def _record_kill(self, stage: str, i: int, sig: int, atr: float, swing: float) -> None:
        """
        Counts a funnel kill and (in diagnostics mode) shadow-simulates the
        killed candidate at zero risk with the normal clamped SL/TP. For the
        sl_clamp stage itself the shadow uses the UNCLAMPED stop, to measure
        what the clamp threw away.
        """
        self.funnel[stage] = self.funnel.get(stage, 0) + 1
        if not self.collect_diagnostics or sig == 0 or not np.isfinite(atr) or atr <= 0:
            return
        entry = self._opens[i] + sig * self._entry_cost(self._opens[i])
        swing_arg = float(swing) if np.isfinite(swing) else None
        sl_tp = self._sl_tp(sig, entry, atr, swing_arg)
        if sl_tp is None:
            if stage != "sl_clamp":
                return  # would have been clamp-skipped anyway — not thrown away
            anchor = swing_arg if swing_arg is not None else entry
            sl = anchor - sig * self.sl_atr_mult * atr
            risk = (entry - sl) * sig
            if risk <= 0:
                return
            tp = entry + sig * self.tp_r * risk
        else:
            sl, tp = sl_tp
            risk = (entry - sl) * sig
        shadow = self._walk_exit(i, sig, entry, sl, tp, risk, use_be=True)
        shadow.update({"kill_stage": stage, "entry_time": self.df.index[i],
                       "direction": sig, "risk_pips": risk / self.pip})
        self.shadow_trades.append(shadow)

    def run(self) -> dict[str, Any]:
        df = self.df
        signal_frame = strategy_module.build_signal_frame(df, self.df_h1, self.params)
        signals = signal_frame["signal"].to_numpy()
        swings = signal_frame["swing_price"].to_numpy()
        atrs = signal_frame["atr_14"].to_numpy()
        atr_medians = signal_frame["atr_median"].to_numpy()

        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        index = df.index
        # cached for _walk_exit / _record_kill
        self._opens, self._highs, self._lows, self._closes = opens, highs, lows, closes
        self._news_bar_mask = self._news_mask(index, config.NEWS_BUFFER_MINS)

        self.funnel = {
            "bars": int(len(df)),
            "regime_bars": int((signal_frame["regime"] != 0).sum()),
            "pullback_bars": int(
                (signal_frame["long_pullback"] | signal_frame["short_pullback"]).sum()
            ),
            "signals": int((signals != 0).sum()),
        }
        self.shadow_trades = []

        balance = self.starting_balance
        self.trades = []
        equity_curve: list[float] = []

        # Pacing state (mirrors RiskManager live behavior)
        day = None
        week = None
        day_start_balance = balance
        week_start_balance = balance
        trades_today = 0
        consec_losses_day = 0
        consec_losses_week = 0
        halted_today = False
        halted_this_week = False

        kill_level = self.starting_balance * (1.0 - config.KILL_SWITCH_PCT)
        kill_switch_hit = False
        kill_switch_time = None
        trading_disabled = False

        open_trade: dict[str, Any] | None = None

        def _close_trade(trade, exit_price, exit_time, reason):
            nonlocal balance, consec_losses_day, consec_losses_week
            d = trade["direction"]
            pnl_pips = (exit_price - trade["entry"]) * d / self.pip
            pnl = pnl_pips * self.pip_value * trade["lot"]
            pnl -= self._commission_per_lot(trade["entry"]) * trade["lot"]
            balance_before = balance
            balance = balance + pnl
            risk = trade["risk_distance"]
            r_mult = ((exit_price - trade["entry"]) * d) / risk
            result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
            if result == "LOSS":
                consec_losses_day += 1
                consec_losses_week += 1
            elif result == "WIN":
                consec_losses_day = 0
                consec_losses_week = 0

            record = {
                **trade,
                "exit": exit_price,
                "exit_time": exit_time,
                "exit_reason": reason,
                "pnl_currency": round(pnl, 2),
                "pct_return": pnl / balance_before,
                "r_multiple": r_mult,
                "result": result,
            }
            if trade["hh"] > -np.inf:
                record["mfe_r"] = ((trade["hh"] - trade["entry"]) if d == 1
                                   else (trade["entry"] - trade["ll"])) / risk
                record["mae_r"] = ((trade["entry"] - trade["ll"]) if d == 1
                                   else (trade["hh"] - trade["entry"])) / risk
            else:
                record["mfe_r"] = record["mae_r"] = 0.0
            if self.collect_diagnostics:
                no_be = self._walk_exit(
                    trade["entry_idx"], d, trade["entry"], trade["sl_initial"],
                    trade["tp"], risk, use_be=False,
                )
                record["no_be_r"] = no_be["r"]
                record["no_be_reason"] = no_be["exit_reason"]
            self.trades.append(record)

        for i in range(1, len(df)):
            ts = index[i]

            # -------- calendar resets (server time == UTC by default) ------
            ts_day = ts.date()
            ts_week = ts.isocalendar()[:2]
            if day != ts_day:
                day = ts_day
                day_start_balance = balance
                trades_today = 0
                consec_losses_day = 0
                halted_today = False
            if week != ts_week:
                week = ts_week
                week_start_balance = balance
                consec_losses_week = 0
                halted_this_week = False

            # -------- manage the open position ----------------------------
            if open_trade is not None:
                d = open_trade["direction"]
                sl = open_trade["sl"]
                tp = open_trade["tp"]
                open_trade["hh"] = max(open_trade["hh"], highs[i])
                open_trade["ll"] = min(open_trade["ll"], lows[i])
                sl_hit = lows[i] <= sl if d == 1 else highs[i] >= sl
                tp_hit = highs[i] >= tp if d == 1 else lows[i] <= tp

                if sl_hit:  # conservative: SL before TP when both touch
                    slip = self._stop_slip(open_trade["entry"], bool(self._news_bar_mask[i]))
                    _close_trade(open_trade, sl - d * slip, ts, "SL")
                    open_trade = None
                elif tp_hit:
                    _close_trade(open_trade, tp, ts, "TP")
                    open_trade = None
                else:
                    if not open_trade["be_moved"]:
                        trigger = open_trade["be_trigger"]
                        reached = highs[i] >= trigger if d == 1 else lows[i] <= trigger
                        if reached:
                            # SL to entry from the NEXT bar (can't know
                            # intra-bar ordering)
                            open_trade["sl"] = open_trade["entry"]
                            open_trade["be_moved"] = True

            # -------- equity & kill switch --------------------------------
            floating = 0.0
            if open_trade is not None:
                d = open_trade["direction"]
                floating = (
                    (closes[i] - open_trade["entry"]) * d / self.pip
                ) * self.pip_value * open_trade["lot"]
            equity = balance + floating
            equity_curve.append(equity)

            if not kill_switch_hit and equity <= kill_level:
                kill_switch_hit = True
                kill_switch_time = ts
                if self.stop_on_kill:
                    trading_disabled = True
                    if open_trade is not None:
                        _close_trade(open_trade, closes[i], ts, "KILL_SWITCH")
                        open_trade = None

            # -------- entry gates ------------------------------------------
            sig = int(signals[i - 1])  # signal at close of bar i-1 → enter at open[i]
            if sig == 0:
                continue
            atr = float(atrs[i - 1])
            swing = float(swings[i - 1])

            if trading_disabled:
                self._record_kill("disabled", i, sig, atr, swing)
                continue
            if open_trade is not None:
                self._record_kill("position_busy", i, sig, atr, swing)
                continue

            if halted_today or halted_this_week:
                self._record_kill("pacing_halted", i, sig, atr, swing)
                continue
            max_trades_today = (
                1 if self.params.entry_mode in ("regime_daily", "range_fade")
                else config.MAX_TRADES_PER_DAY  # regime_daily2 uses the 2/day cap
            )
            if trades_today >= max_trades_today:
                self._record_kill("pacing_trades_per_day", i, sig, atr, swing)
                continue
            if consec_losses_day >= config.MAX_CONSEC_LOSSES_DAY:
                halted_today = True
                self._record_kill("pacing_consec_day", i, sig, atr, swing)
                continue
            if consec_losses_week >= config.MAX_CONSEC_LOSSES_WEEK:
                halted_this_week = True
                self._record_kill("pacing_consec_week", i, sig, atr, swing)
                continue
            if day_start_balance > 0 and (day_start_balance - equity) / day_start_balance >= config.DAILY_LOSS_PCT:
                halted_today = True
                self._record_kill("pacing_daily_loss", i, sig, atr, swing)
                continue
            if week_start_balance > 0 and (week_start_balance - equity) / week_start_balance >= config.WEEKLY_STOP_PCT:
                halted_this_week = True
                self._record_kill("pacing_weekly_loss", i, sig, atr, swing)
                continue

            if not strategy_module.entry_session_ok(ts):
                self._record_kill("session", i, sig, atr, swing)
                continue
            hour = ts.hour
            if config.ROLLOVER_START_UTC <= hour < config.ROLLOVER_END_UTC:
                self._record_kill("rollover", i, sig, atr, swing)
                continue
            if self._in_news_window(ts, config.NEWS_MAJOR_BUFFER_MINS):
                self._record_kill("news", i, sig, atr, swing)
                continue

            atr_median = float(atr_medians[i - 1]) if np.isfinite(atr_medians[i - 1]) else None
            atr_pips = atr / self.pip
            if not np.isfinite(atr_pips) or not self._atr_floor_ok(atr, opens[i]):
                self._record_kill("vol_floor", i, sig, atr, swing)
                continue
            if atr_median and atr_pips > config.ATR_MAX_MEDIAN_MULT * (atr_median / self.pip):
                self._record_kill("vol_ceiling", i, sig, atr, swing)
                continue

            # -------- entry --------------------------------------------------
            raw_open = opens[i]
            entry = raw_open + sig * self._entry_cost(raw_open)
            swing_arg = float(swing) if np.isfinite(swing) else None
            sl_tp = self._sl_tp(sig, entry, atr, swing_arg)
            if sl_tp is None:
                self._record_kill("sl_clamp", i, sig, atr, swing)
                continue
            sl, tp = sl_tp
            risk_distance = (entry - sl) if sig == 1 else (sl - entry)
            lot = self._lot_size(balance, risk_distance)
            if lot <= 0:
                self._record_kill("lot_zero", i, sig, atr, swing)
                continue

            self.funnel["executed"] = self.funnel.get("executed", 0) + 1
            be_trigger = (
                entry + sig * self.be_at_r * risk_distance
                if self.be_at_r is not None else None
            )
            open_trade = {
                "direction": sig,
                "entry_idx": i,
                "entry_time": ts,
                "entry": entry,
                "sl": sl,
                "sl_initial": sl,   # "sl" moves to entry on the BE trigger
                "tp": tp,
                "lot": lot,
                "risk_distance": risk_distance,
                "be_trigger": be_trigger,
                "be_moved": self.be_at_r is None,
                "hh": -np.inf,
                "ll": np.inf,
            }
            trades_today += 1

        # close any trailing open trade at the last close
        if open_trade is not None:
            _close_trade(open_trade, closes[-1], index[-1], "END_OF_DATA")
            open_trade = None

        pct_returns = np.array([t["pct_return"] for t in self.trades], dtype=float)
        r_multiples = np.array([t["r_multiple"] for t in self.trades], dtype=float)
        from monte_carlo_dd import trade_stats as _stats

        metrics = _stats(pct_returns, r_multiples)
        metrics.update(
            {
                "ending_balance": round(balance, 2),
                "total_return_pct": round(
                    (balance - self.starting_balance) / self.starting_balance * 100.0, 2
                ),
                "kill_switch_hit": kill_switch_hit,
                "kill_switch_time": str(kill_switch_time) if kill_switch_time else None,
                "start": str(index[0]),
                "end": str(index[-1]),
                "funnel": dict(self.funnel),
            }
        )
        return metrics

    def pct_returns(self) -> np.ndarray:
        return np.array([t["pct_return"] for t in self.trades], dtype=float)

    def r_multiples(self) -> np.ndarray:
        return np.array([t["r_multiple"] for t in self.trades], dtype=float)


def _print_rules_metrics(metrics: dict[str, Any]) -> None:
    print("\nRULES-STRATEGY BACKTEST (after costs)")
    print("-" * 56)
    print(f"Period:                {metrics['start']} → {metrics['end']}")
    print(f"Trades:                {metrics['trades']}")
    print(f"Total return:          {metrics['total_return_pct']:.2f}%")
    print(f"Profit factor:         {metrics['profit_factor']}")
    print(f"Win rate:              {metrics['win_rate_pct']:.2f}%")
    print(f"Avg R:                 {metrics['avg_r']}")
    print(f"Expectancy/trade:      {metrics['expectancy_pct']:.4f}%")
    print(f"Max drawdown:          {metrics['max_drawdown_pct']:.2f}%")
    print(f"Longest losing streak: {metrics['longest_losing_streak']}")
    print(f"Kill switch (−3%):     {'HIT at ' + str(metrics['kill_switch_time']) if metrics['kill_switch_hit'] else 'never hit'}")


def run_go_no_go(fast: bool = False) -> dict[str, Any]:
    """
    Single-command GO/NO-GO evaluation:
      1. full-history rules backtest with costs
      2. 24m/6m walk-forward (6m step) incl. parameter perturbation
      3. ≥20,000 block-bootstrap Monte Carlo paths through the Bootcamp
         step simulator
      4. verdict: GO only if P(breach −5%) < 1%, P(kill) < 10%,
         P(pass) > 70%, and PF ≥ 1.25 after costs in EVERY fold.
    """
    from monte_carlo_dd import print_step_report, run_step_monte_carlo
    from validation import walk_forward_rules

    df_m15 = get_ohlcv_from_csv("EURUSD", "M15")
    df_h1 = get_ohlcv_from_csv("EURUSD", "H1")

    engine = RulesBacktestEngine(df_m15, df_h1)
    metrics = engine.run()
    _print_rules_metrics(metrics)

    if metrics["trades"] < 30:
        print("\n⛔ NO-GO: fewer than 30 trades in the full backtest — no basis for inference.")
        return {"verdict": "NO-GO", "reason": "insufficient trades", "metrics": metrics}

    folds = walk_forward_rules(df_m15, df_h1, perturb=not fast)
    print("\nWALK-FORWARD (24m train / 6m test, 6m step)")
    print("-" * 56)
    all_folds_pf_ok = True
    for fold in folds:
        pf = fold["profit_factor"]
        ok = (pf == float("inf")) or (pf >= 1.25)
        if fold["trades"] < 5:
            ok = False  # too few trades to trust the fold
        all_folds_pf_ok &= ok
        perturbed = fold.get("perturbed_pf_range")
        print(
            f"Fold {fold['fold']}: test {fold['test_start']}→{fold['test_end']}  "
            f"trades={fold['trades']}  PF={pf}  ret={fold['total_return_pct']:.2f}%  "
            f"maxDD={fold['max_drawdown_pct']:.2f}%"
            + (f"  perturbed PF {perturbed[0]:.2f}–{perturbed[1]:.2f}" if perturbed else "")
            + ("  ✓" if ok else "  ✗")
        )

    n_paths = 2_000 if fast else 20_000
    mc = run_step_monte_carlo(engine.pct_returns(), engine.r_multiples(), n_paths=n_paths)
    print_step_report(mc)

    checks = {
        "P(breach −5%) < 1%": mc["p_breach_official"] < 0.01,
        "P(kill switch) < 10%": mc["p_kill_switch"] < 0.10,
        "P(pass) > 70%": mc["p_pass"] > 0.70,
        "PF ≥ 1.25 in every fold": all_folds_pf_ok,
    }
    verdict = "GO" if all(checks.values()) else "NO-GO"

    print("\n" + "=" * 56)
    print("GO/NO-GO VERDICT")
    print("=" * 56)
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print(f"\n  ➜ {verdict}" + ("" if verdict == "GO" else "  — do not run this on a challenge account."))

    return {"verdict": verdict, "checks": checks, "metrics": metrics, "monte_carlo": mc, "folds": folds}


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
    parser = argparse.ArgumentParser(description="The5ers Bootcamp backtests")
    parser.add_argument(
        "--go-no-go", action="store_true",
        help="Full GO/NO-GO evaluation: rules backtest + walk-forward + Monte Carlo.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced Monte Carlo paths / no perturbation (smoke test only).",
    )
    parser.add_argument(
        "--legacy-model", action="store_true",
        help="Run the legacy ML-model backtest instead of the rules strategy.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("risk_manager").setLevel(logging.ERROR)

    if not args.legacy_model:
        if args.go_no_go:
            run_go_no_go(fast=args.fast)
        else:
            df_m15 = get_ohlcv_from_csv("EURUSD", "M15")
            df_h1 = get_ohlcv_from_csv("EURUSD", "H1")
            engine = RulesBacktestEngine(df_m15, df_h1)
            _print_rules_metrics(engine.run())
        return

    _legacy_model_main()


def _legacy_model_main() -> None:
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
