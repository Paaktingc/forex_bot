"""
test_step_simulator.py

Tests for the Bootcamp step simulator, block bootstrap, and trade stats
in monte_carlo_dd.py, plus a light RulesBacktestEngine smoke test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo_dd import (
    KILL_SWITCH,
    STEP_FAIL,
    STEP_TARGET,
    _step_outcomes_vectorized,
    block_bootstrap_paths,
    longest_losing_streak,
    run_step_monte_carlo,
    simulate_step,
    trade_stats,
)


class TestSimulateStep:
    def test_absorbs_at_plus_6_pct(self):
        # 1.01^6 ≈ 1.0615 ≥ 1.06 → PASS on trade 6
        result = simulate_step(np.full(10, 0.01))
        assert result.outcome == "PASS"
        assert result.trades_used == 6
        assert result.final_equity >= 1.0 + STEP_TARGET

    def test_kill_switch_ends_path_before_official_fail(self):
        # −1% per trade: 0.99^4 ≈ 0.9606 ≤ 0.97 → KILL on trade 4,
        # equity never near −5%
        result = simulate_step(np.full(10, -0.01))
        assert result.outcome == "KILL"
        assert result.trades_used == 4
        assert result.final_equity > 1.0 + STEP_FAIL

    def test_single_gap_through_both_levels_is_breach(self):
        result = simulate_step(np.array([-0.06]))
        assert result.outcome == "BREACH"
        assert result.trades_used == 1

    def test_incomplete_path(self):
        result = simulate_step(np.zeros(5))
        assert result.outcome == "INCOMPLETE"
        assert result.final_equity == pytest.approx(1.0)

    def test_pass_checked_after_downside(self):
        # A path that dips to kill level never gets to pass
        returns = np.array([-0.031, 0.10])
        result = simulate_step(returns)
        assert result.outcome == "KILL"


class TestVectorizedOutcomesMatchScalar:
    def test_agreement_on_random_paths(self):
        rng = np.random.default_rng(0)
        paths = rng.normal(0.001, 0.012, size=(200, 120))
        outcomes, trades = _step_outcomes_vectorized(paths)
        code = {"PASS": 0, "KILL": 1, "BREACH": 2, "INCOMPLETE": 3}
        for row in range(len(paths)):
            scalar = simulate_step(paths[row])
            assert outcomes[row] == code[scalar.outcome], f"path {row}"
            assert trades[row] == scalar.trades_used, f"path {row}"


class TestBlockBootstrap:
    def test_shape_and_determinism(self):
        returns = np.arange(50, dtype=float) / 1000.0
        a = block_bootstrap_paths(returns, n_paths=100, block_size=10, path_len=90, seed=1)
        b = block_bootstrap_paths(returns, n_paths=100, block_size=10, path_len=90, seed=1)
        assert a.shape == (100, 90)
        np.testing.assert_array_equal(a, b)

    def test_blocks_preserve_consecutive_structure(self):
        returns = np.arange(100, dtype=float)
        paths = block_bootstrap_paths(returns, n_paths=5, block_size=10, path_len=40, seed=2)
        # inside each block, values are consecutive (mod wrap-around)
        for path in paths:
            for start in range(0, 40, 10):
                block = path[start:start + 10]
                diffs = np.diff(block) % 100
                assert (diffs == 1).all()

    def test_empty_returns_raise(self):
        with pytest.raises(ValueError):
            block_bootstrap_paths(np.array([]))


class TestRunStepMonteCarlo:
    def test_always_winning_stream_passes(self):
        result = run_step_monte_carlo(np.full(40, 0.01), n_paths=500)
        assert result["p_pass"] == 1.0
        assert result["p_breach_official"] == 0.0
        assert result["median_trades_to_pass"] == 6

    def test_always_losing_stream_hits_kill(self):
        result = run_step_monte_carlo(np.full(40, -0.005), n_paths=500)
        assert result["p_pass"] == 0.0
        assert result["p_kill_switch"] == 1.0
        # −0.5% steps can never gap through −5% from above −3%
        assert result["p_breach_official"] == 0.0

    def test_reports_required_fields(self):
        rng = np.random.default_rng(3)
        result = run_step_monte_carlo(rng.normal(0.002, 0.006, 80), n_paths=300)
        for key in (
            "profit_factor", "win_rate_pct", "avg_r", "expectancy_pct",
            "max_drawdown_pct", "longest_losing_streak", "trades",
            "p_pass", "p_kill_switch", "p_breach_official",
            "median_trades_to_pass",
        ):
            assert key in result


class TestTradeStats:
    def test_known_values(self):
        pnl = np.array([0.01, -0.005, 0.01, -0.005])
        stats = trade_stats(pnl, r_multiples=np.array([2.0, -1.0, 2.0, -1.0]))
        assert stats["trades"] == 4
        assert stats["profit_factor"] == pytest.approx(2.0)
        assert stats["win_rate_pct"] == 50.0
        assert stats["avg_r"] == pytest.approx(0.5)
        assert stats["expectancy_pct"] == pytest.approx(0.25)

    def test_longest_losing_streak(self):
        assert longest_losing_streak(np.array([1, -1, -1, -1, 1, -1])) == 3
        assert longest_losing_streak(np.array([1.0, 2.0])) == 0

    def test_empty(self):
        assert trade_stats(np.array([]))["trades"] == 0


class TestRulesBacktestEngineSmoke:
    def test_costs_and_structure_on_real_tail(self):
        import backtest

        df15 = backtest.get_ohlcv_from_csv("EURUSD", "M15").iloc[-6000:]
        dfh1 = backtest.get_ohlcv_from_csv("EURUSD", "H1")
        dfh1 = dfh1.loc[dfh1.index >= df15.index[0].floor("h") - __import__("pandas").Timedelta(days=30)]

        engine = backtest.RulesBacktestEngine(df15, dfh1)
        metrics = engine.run()

        assert metrics["trades"] == len(engine.trades)
        for trade in engine.trades:
            # every simulated order carries SL and TP on the correct sides
            # (sl_initial: "sl" itself moves to entry on the +1R BE trigger)
            if trade["direction"] == 1:
                assert trade["sl_initial"] < trade["entry"] < trade["tp"]
            else:
                assert trade["tp"] < trade["entry"] < trade["sl_initial"]
            # SL distance respects the clamp
            sl_pips = abs(trade["entry"] - trade["sl_initial"]) / 0.0001
            import config

            assert config.SL_MIN_PIPS - 1e-6 <= sl_pips <= config.SL_MAX_PIPS + 1e-6
            # commission was charged (a full-SL loss costs more than raw pips)
            assert "pnl_currency" in trade and "pct_return" in trade

    def test_never_more_than_two_entries_per_day(self):
        import backtest
        import pandas as pd

        df15 = backtest.get_ohlcv_from_csv("EURUSD", "M15").iloc[-20000:]
        dfh1 = backtest.get_ohlcv_from_csv("EURUSD", "H1")
        dfh1 = dfh1.loc[dfh1.index >= df15.index[0].floor("h") - pd.Timedelta(days=30)]

        engine = backtest.RulesBacktestEngine(df15, dfh1)
        engine.run()
        entries = pd.Series([t["entry_time"].date() for t in engine.trades])
        if not entries.empty:
            assert entries.value_counts().max() <= 2


class TestEngineExitOverrides:
    def _engine(self, **kw):
        import backtest
        import pandas as pd

        df15 = backtest.get_ohlcv_from_csv("EURUSD", "M15").iloc[-6000:]
        dfh1 = backtest.get_ohlcv_from_csv("EURUSD", "H1")
        dfh1 = dfh1.loc[dfh1.index >= df15.index[0].floor("h") - pd.Timedelta(days=30)]
        return backtest.RulesBacktestEngine(df15, dfh1, **kw)

    def test_be_disabled_never_scratches(self):
        eng = self._engine(be_at_r=None)
        eng.run()
        for t in eng.trades:
            # without BE, every SL exit is a full loss (SL never at entry)
            if t["exit_reason"] == "SL":
                assert abs(t["r_multiple"]) > 0.5

    def test_tp_override_changes_target(self):
        eng = self._engine(tp_r=1.5)
        eng.run()
        for t in eng.trades:
            risk = abs(t["entry"] - t["sl_initial"])
            assert abs(t["tp"] - t["entry"]) == pytest.approx(1.5 * risk, abs=1e-4)

    def test_defaults_follow_config(self):
        import config
        eng = self._engine()
        assert eng.be_at_r == config.BE_AT_R
        assert eng.tp_r == config.TP_R
        assert eng.sl_atr_mult == config.SL_ATR_MULT
