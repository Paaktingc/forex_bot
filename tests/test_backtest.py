"""
test_backtest.py

Unit tests for the backtest.py module.
"""

from itertools import count
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))

import backtest


def make_frames(rows: int = 140, base_close: float = 1.1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01 00:00", periods=rows, freq="15min", tz="UTC")
    df_m15 = pd.DataFrame(
        {
            "open": np.full(rows, base_close),
            "high": np.full(rows, base_close + 0.0005),
            "low": np.full(rows, base_close - 0.0005),
            "close": np.full(rows, base_close),
            "volume": np.full(rows, 1000),
        },
        index=index,
    )

    h1_rows = max(80, rows // 4 + 10)
    h1_index = pd.date_range(index[0].floor("h"), periods=h1_rows, freq="1h", tz="UTC")
    df_h1 = pd.DataFrame(
        {"close": np.linspace(base_close, base_close + 0.0100, h1_rows)},
        index=h1_index,
    )
    return df_m15, df_h1


@pytest.fixture
def dummy_artifacts(monkeypatch):
    monkeypatch.setattr(backtest, "load_model", lambda: ("model", "encoder"))


def test_backtest_engine_initialization_normalizes_inputs(dummy_artifacts):
    df_m15, df_h1 = make_frames()
    df_m15 = df_m15.sort_index(ascending=False)
    df_h1 = df_h1.sort_index(ascending=False)

    engine = backtest.BacktestEngine(df_m15, df_h1, starting_balance=12_500)

    assert engine.balance == 12_500
    assert engine.starting_balance == 12_500
    assert str(engine.df.index.tz) == "UTC"
    assert engine.df.index.is_monotonic_increasing
    assert engine.model == "model"
    assert engine.label_encoder == "encoder"


def test_backtest_engine_validation_rejects_missing_columns(dummy_artifacts):
    df_m15, df_h1 = make_frames()
    with pytest.raises(ValueError):
        backtest.BacktestEngine(df_m15.drop(columns=["low"]), df_h1)


def test_run_builds_features_with_strict_prefix_only(dummy_artifacts, monkeypatch):
    df_m15, df_h1 = make_frames(rows=105)
    engine = backtest.BacktestEngine(df_m15, df_h1)
    feature_index = df_m15.index
    engine.feature_frame = pd.DataFrame(
        {"atr_14": np.full(len(feature_index), 0.0010)},
        index=feature_index,
    )

    seen_rows = []

    def fake_predict_signal(model, le, X_live):
        seen_rows.append(X_live.index[0])
        return (0, 0.0)

    monkeypatch.setattr(backtest, "predict_signal", fake_predict_signal)

    metrics = engine.run()

    assert seen_rows[0] == df_m15.index[100]
    assert metrics["total_trades"] == 0


@pytest.mark.parametrize(
    ("exit_bar", "bar_updates", "expected_reason", "expected_result"),
    [
        # ATR = 10 pips → SL = 1.5×ATR = 15 pips below entry, TP = 2R = 30 pips above
        (
            102,
            {"high": 1.1035, "low": 1.1005, "close": 1.1010},
            "TP",
            "WIN",
        ),
        (
            102,
            {"high": 1.1005, "low": 1.0985, "close": 1.0990},
            "SL",
            "LOSS",
        ),
        (
            126,
            {"high": 1.1010, "low": 1.0995, "close": 1.1003},
            "TIME_LIMIT",
            "BE",
        ),
    ],
)
def test_run_closes_trades_on_tp_sl_and_time_limit(
    dummy_artifacts,
    monkeypatch,
    exit_bar,
    bar_updates,
    expected_reason,
    expected_result,
):
    df_m15, df_h1 = make_frames(rows=130)
    for bar in range(101, exit_bar + 1):
        for column, value in bar_updates.items():
            df_m15.iloc[bar, df_m15.columns.get_loc(column)] = value

    engine = backtest.BacktestEngine(df_m15, df_h1)
    engine.feature_frame = pd.DataFrame(
        {"atr_14": np.full(len(df_m15), 0.0010)},
        index=df_m15.index,
    )

    calls = count()

    def fake_predict_signal(model, le, X_live):
        return (1, 0.90) if next(calls) == 0 else (0, 0.0)

    monkeypatch.setattr(backtest, "predict_signal", fake_predict_signal)

    metrics = engine.run()

    assert metrics["total_trades"] == 1
    trade = engine.trades[0]
    assert trade["exit_reason"] == expected_reason
    assert trade["result"] == expected_result
    assert trade["duration_candles"] == exit_bar - 101


def test_run_skips_entries_during_rollover_or_news_windows(dummy_artifacts, monkeypatch):
    df_m15, df_h1 = make_frames(rows=110)
    engine = backtest.BacktestEngine(df_m15, df_h1)
    engine.feature_frame = pd.DataFrame(
        {"atr_14": np.full(len(df_m15), 0.0010)},
        index=df_m15.index,
    )
    monkeypatch.setattr(backtest, "predict_signal", lambda model, le, X_live: (1, 0.90))
    monkeypatch.setattr(engine, "_historical_rollover_window", lambda ts: True)
    monkeypatch.setattr(engine, "_historical_news_window", lambda ts: False)

    rollover_metrics = engine.run()
    assert rollover_metrics["total_trades"] == 0

    monkeypatch.setattr(engine, "_historical_rollover_window", lambda ts: False)
    monkeypatch.setattr(engine, "_historical_news_window", lambda ts: True)

    news_metrics = engine.run()
    assert news_metrics["total_trades"] == 0


def test_run_respects_max_open_trades_limit(dummy_artifacts, monkeypatch):
    df_m15, df_h1 = make_frames(rows=115)
    engine = backtest.BacktestEngine(df_m15, df_h1)
    engine.feature_frame = pd.DataFrame(
        {"atr_14": np.full(len(df_m15), 0.0010)},
        index=df_m15.index,
    )
    monkeypatch.setattr(backtest, "predict_signal", lambda model, le, X_live: (1, 0.90))
    monkeypatch.setattr(backtest.config, "MAX_OPEN_TRADES", 1)

    metrics = engine.run()

    assert metrics["total_trades"] == 1


def test_compute_metrics_returns_expected_values(dummy_artifacts):
    df_m15, df_h1 = make_frames()
    engine = backtest.BacktestEngine(df_m15, df_h1)
    engine.trades = [
        {"pnl_currency": 50.0, "result": "WIN", "duration_candles": 5},
        {"pnl_currency": -25.0, "result": "LOSS", "duration_candles": 10},
        {"pnl_currency": 0.0, "result": "BE", "duration_candles": 15},
    ]

    metrics = engine.compute_metrics([10_000.0, 10_050.0, 10_025.0, 10_025.0])

    assert metrics["total_return_pct"] == 0.25
    assert metrics["max_drawdown_pct"] == pytest.approx(0.25, abs=0.01)
    assert metrics["profit_factor"] == 2.0
    assert metrics["win_rate_pct"] == pytest.approx(33.33, abs=0.01)
    assert metrics["total_trades"] == 3
    assert metrics["avg_trade_duration_candles"] == 10.0
    assert metrics["sharpe_ratio"] > 0


def test_walk_forward_validation_returns_five_windows_plus_average(dummy_artifacts, monkeypatch):
    df_m15, df_h1 = make_frames(rows=300)
    engine = backtest.BacktestEngine(df_m15, df_h1)

    def fake_prepare_frame(self, train_m15, train_h1):
        frame = train_m15[["open", "high", "low", "close"]].copy()
        frame["atr_14"] = 0.0010
        frame["feature_a"] = 1.0
        return frame

    def fake_apply_triple_barrier(frame):
        labelled = frame.copy()
        labels = np.where(np.arange(len(labelled)) % 2 == 0, 1, -1)
        labelled["label"] = labels
        return labelled

    def fake_prepare_training_data(frame):
        X = frame[["feature_a"]].copy()
        y = np.where(np.arange(len(frame)) % 2 == 0, 0, 1)
        le = LabelEncoder()
        le.fit([-1, 1])
        return X, y, le

    run_counter = count(1)

    def fake_run(self):
        idx = next(run_counter)
        metrics = {
            "total_return_pct": float(idx),
            "max_drawdown_pct": 1.0,
            "sharpe_ratio": 1.5,
            "profit_factor": 2.0,
            "win_rate_pct": 55.0,
            "total_trades": 60,
            "avg_trade_duration_candles": 8.0,
        }
        metrics.update(backtest._evaluate_pass_criteria(metrics))
        return metrics

    monkeypatch.setattr(backtest.BacktestEngine, "_prepare_training_frame", fake_prepare_frame)
    monkeypatch.setattr(backtest, "apply_triple_barrier", fake_apply_triple_barrier)
    monkeypatch.setattr(backtest, "prepare_training_data", fake_prepare_training_data)
    monkeypatch.setattr(backtest, "train_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(backtest.BacktestEngine, "run", fake_run)

    results = engine.walk_forward_validation()

    assert len(results) == 6
    assert results[0]["window"] == 1
    assert results[-1]["window"] == "average"
    assert results[-1]["total_return_pct"] == 3.0


def test_main_logs_error_when_csvs_are_missing(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(backtest.config, "DATA_DIR", tmp_path / "missing_data")
    caplog.set_level("ERROR")

    backtest.main()

    assert "Failed to load historical CSVs" in caplog.text
