"""
test_pooling.py

Tests for research_pooling.pool_trades (global one-open-trade + pacing)
and the engine's multi-symbol support.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from research_pooling import pool_trades

UTC = "UTC"


def _trade(entry, exit_, pct, symbol="EURUSD", reason="TP"):
    return {
        "entry_time": pd.Timestamp(entry, tz=UTC),
        "exit_time": pd.Timestamp(exit_, tz=UTC),
        "pct_return": pct,
        "exit_reason": reason,
        "symbol": symbol,
    }


class TestPoolTrades:
    def test_overlapping_trade_dropped(self):
        trades = [
            _trade("2024-01-02 08:00", "2024-01-02 14:00", 0.006, "EURUSD"),
            _trade("2024-01-02 09:00", "2024-01-02 11:00", 0.006, "GBPUSD"),
        ]
        pooled = pool_trades(trades)
        assert len(pooled) == 1
        assert pooled.iloc[0].symbol == "EURUSD"  # earlier entry wins

    def test_non_overlapping_both_kept(self):
        trades = [
            _trade("2024-01-02 08:00", "2024-01-02 10:00", 0.006, "EURUSD"),
            _trade("2024-01-02 11:00", "2024-01-02 12:00", 0.006, "GBPUSD"),
        ]
        assert len(pool_trades(trades)) == 2

    def test_global_two_trades_per_day_cap(self):
        trades = [
            _trade("2024-01-02 08:00", "2024-01-02 08:30", 0.006, "AUDUSD"),
            _trade("2024-01-02 09:00", "2024-01-02 09:30", 0.006, "EURUSD"),
            _trade("2024-01-02 10:00", "2024-01-02 10:30", 0.006, "GBPUSD"),
        ]
        pooled = pool_trades(trades)
        assert len(pooled) == config.MAX_TRADES_PER_DAY == 2
        # next day is fresh
        trades.append(_trade("2024-01-03 08:00", "2024-01-03 09:00", 0.006, "GBPUSD"))
        assert len(pool_trades(trades)) == 3

    def test_two_consecutive_losses_halt_day(self):
        trades = [
            _trade("2024-01-02 08:00", "2024-01-02 08:30", -0.003, "AUDUSD"),
            _trade("2024-01-02 09:00", "2024-01-02 09:30", -0.003, "EURUSD"),
            _trade("2024-01-02 10:00", "2024-01-02 10:30", 0.006, "GBPUSD"),
        ]
        pooled = pool_trades(trades)
        assert len(pooled) == 2  # third blocked by consec-loss halt
        assert (pooled.pct_return < 0).all()

    def test_symbol_tiebreak_deterministic(self):
        trades = [
            _trade("2024-01-02 08:00", "2024-01-02 09:00", 0.006, "GBPUSD"),
            _trade("2024-01-02 08:00", "2024-01-02 09:00", 0.006, "AUDUSD"),
        ]
        pooled = pool_trades(trades)
        assert len(pooled) == 1
        assert pooled.iloc[0].symbol == "AUDUSD"  # alphabetical


class TestEngineSymbolSupport:
    def test_usdjpy_pip_scale(self):
        import backtest

        df15 = backtest.get_ohlcv_from_csv("USDJPY", "M15").iloc[-3000:]
        dfh1 = backtest.get_ohlcv_from_csv("USDJPY", "H1")
        dfh1 = dfh1.loc[dfh1.index >= df15.index[0].floor("h") - pd.Timedelta(days=30)]
        eng = backtest.RulesBacktestEngine(df15, dfh1, symbol="USDJPY")
        assert eng.pip == 0.01
        assert eng.spread == pytest.approx(
            config.BACKTEST_SPREAD_FLOOR_BY_SYMBOL["USDJPY"] * 0.01
        )
        eng.run()
        for t in eng.trades:
            sl_pips = abs(t["entry"] - t["sl_initial"]) / 0.01
            assert config.SL_MIN_PIPS - 1e-6 <= sl_pips <= config.SL_MAX_PIPS + 1e-6

    def test_unknown_symbol_rejected(self):
        import backtest

        df15 = backtest.get_ohlcv_from_csv("EURUSD", "M15").iloc[-500:]
        dfh1 = backtest.get_ohlcv_from_csv("EURUSD", "H1").iloc[-500:]
        with pytest.raises(ValueError):
            backtest.RulesBacktestEngine(df15, dfh1, symbol="NZDCAD")
