"""
backtest.py

Historical candle-by-candle backtesting for the THE5ERS forex bot.
Implements strict no-lookahead feature generation, trade simulation,
summary metrics, and walk-forward validation.
"""

from __future__ import annotations

import logging
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
from model import load_model, predict_signal, train_model
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

ENTRY_SPREAD = 0.00012
PIP_SIZE = 0.0001
PIP_VALUE_PER_LOT = 10.0
ANNUALIZATION_FACTOR = sqrt(252 * 24 * 4)

PASS_CRITERIA = {
    "total_return": ("Total return", 8.0, lambda value: value > 8.0, "> 8%"),
    "max_drawdown": ("Max drawdown", 4.0, lambda value: value < 4.0, "< 4%"),
    "win_rate": ("Win rate", 45.0, lambda value: value > 45.0, "> 45%"),
    "profit_factor": ("Profit factor", 1.3, lambda value: value > 1.3, "> 1.3"),
    "sharpe_ratio": ("Sharpe ratio", 0.8, lambda value: value > 0.8, "> 0.8"),
    "min_trades": ("Min trades", 50, lambda value: value > 50, "> 50"),
}


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


def _print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)
    print(f"Total Return %           : {metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown %           : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio             : {metrics['sharpe_ratio']:.2f}")
    print(f"Profit Factor            : {metrics['profit_factor']:.2f}")
    print(f"Win Rate %               : {metrics['win_rate_pct']:.2f}%")
    print(f"Total Trades             : {metrics['total_trades']}")
    print(f"Avg Trade Duration Bars  : {metrics['avg_trade_duration_candles']:.2f}")
    print("-" * 56)
    for key, (label, _, _, target_text) in PASS_CRITERIA.items():
        passed = metrics.get("pass_criteria", {}).get(key, False)
        status = "PASS" if passed else "FAIL"
        print(f"{'✅' if passed else '❌'} {label:<16} {status:<4} ({_format_value(key, metrics)} {target_text})")


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
        current_time = self.df.index[bar_idx]
        df_m15_slice = self.df.iloc[: bar_idx + 1].copy()
        df_h1_slice = self.df_h1.loc[self.df_h1.index <= current_time].copy()
        if df_h1_slice.empty:
            return None

        feature_frame = build_feature_matrix(df_m15_slice, df_h1_slice)
        if feature_frame.empty or current_time not in feature_frame.index:
            return None

        return feature_frame.loc[current_time]

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
        entry_price = round(float(self.df["close"].iloc[bar_idx]) + ENTRY_SPREAD, 5)
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

        for bar_idx in range(100, len(self.df)):
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

        for window, (train_idx, test_idx) in enumerate(splitter.split(self.df), start=1):
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

                fold_model = train_model(X_train, y_encoded)
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    m15_path = config.DATA_DIR / "EURUSD_M15.csv"
    h1_path = config.DATA_DIR / "EURUSD_H1.csv"

    try:
        df_m15 = _load_price_csv(m15_path)
        df_h1 = _load_price_csv(h1_path)
    except Exception as exc:
        logger.error("Failed to load historical CSVs: %s", exc)
        return

    try:
        engine = BacktestEngine(df_m15, df_h1)
        metrics = engine.run()
        walk_forward_results = engine.walk_forward_validation()
    except Exception as exc:
        logger.error("Backtest execution failed: %s", exc)
        return

    _print_metric_block("BACKTEST ENGINE RESULTS", metrics)

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

    _print_metric_block("WALK-FORWARD AVERAGE", walk_forward_results[-1])


if __name__ == "__main__":
    main()
