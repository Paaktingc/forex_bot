"""
test_strategy.py

Tests for the rules-based signal engine: H1 regime, M15 pullback trigger,
and the spread / volatility / session entry filters (incl. London DST).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import strategy
from strategy import (
    Candidate,
    StrategyParams,
    build_signal_frame,
    entry_session_ok,
    generate_candidate,
    h1_regime,
    spread_ok,
    volatility_ok,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_h1(prices: np.ndarray, start="2025-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0005,
            "low": prices - 0.0005,
            "close": prices,
            "volume": 1000,
        },
        index=index,
    )


def _make_m15(prices: np.ndarray, start="2025-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0003,
            "low": prices - 0.0003,
            "close": prices,
            "volume": 500,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# H1 regime
# ---------------------------------------------------------------------------

class TestH1Regime:
    def test_uptrend_gives_long_regime(self):
        prices = np.linspace(1.05, 1.15, 400)  # strong steady uptrend
        regime = h1_regime(_make_h1(prices), StrategyParams(use_adx_gate=False))
        assert regime.iloc[-1] == 1

    def test_downtrend_gives_short_regime(self):
        prices = np.linspace(1.15, 1.05, 400)
        regime = h1_regime(_make_h1(prices), StrategyParams(use_adx_gate=False))
        assert regime.iloc[-1] == -1

    def test_flat_market_gives_no_regime(self):
        # Perfectly flat close → EMA50 == EMA200 == close → strict
        # inequalities fail → no regime anywhere
        prices = np.full(400, 1.10)
        regime = h1_regime(_make_h1(prices), StrategyParams(use_adx_gate=False))
        assert (regime == 0).all()

    def test_adx_gate_blocks_weak_trend(self):
        prices = np.linspace(1.05, 1.15, 400)
        ungated = h1_regime(_make_h1(prices), StrategyParams(use_adx_gate=False))
        assert (ungated == 1).any()
        # An unreachable ADX threshold must gate every bar to no-regime
        gated = h1_regime(
            _make_h1(prices), StrategyParams(use_adx_gate=True, adx_min=101.0)
        )
        assert (gated == 0).all()

    def test_warmup_has_no_regime(self):
        prices = np.linspace(1.05, 1.15, 400)
        regime = h1_regime(_make_h1(prices), StrategyParams(use_adx_gate=False))
        assert (regime.iloc[:199] == 0).all()  # EMA200 not formed yet


# ---------------------------------------------------------------------------
# Entry trigger
# ---------------------------------------------------------------------------

def _trending_market_with_pullback() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds an H1 uptrend and an M15 series that rides above EMA20, dips to
    touch it, then closes back above with an RSI recross.
    """
    h1_prices = np.linspace(1.0500, 1.1500, 500)
    df_h1 = _make_h1(h1_prices, start="2025-01-01")

    n = 600
    base = np.linspace(1.1300, 1.1500, n)  # uptrend, keeps price > EMA20 mostly
    wave = 0.0012 * np.sin(np.arange(n) / 6.0)  # regular pullbacks through EMA20
    m15_prices = base + wave
    start = df_h1.index[-1] - pd.Timedelta(minutes=15 * (n - 1))
    df_m15 = _make_m15(m15_prices, start=start)
    return df_m15, df_h1


class TestEntryTrigger:
    def test_pullback_recross_generates_long_signals(self):
        df_m15, df_h1 = _trending_market_with_pullback()
        frame = build_signal_frame(
            df_m15, df_h1,
            StrategyParams(use_adx_gate=False, entry_mode="pullback_rsi"),
        )
        signals = frame["signal"]
        assert (signals == 1).any(), "expected at least one long trigger"
        assert not (signals == -1).any(), "no shorts in a long regime"

    def test_signal_carries_swing_and_atr(self):
        df_m15, df_h1 = _trending_market_with_pullback()
        frame = build_signal_frame(
            df_m15, df_h1,
            StrategyParams(use_adx_gate=False, entry_mode="pullback_rsi"),
        )
        fired = frame[frame["signal"] == 1]
        assert np.isfinite(fired["swing_price"]).all()
        assert np.isfinite(fired["atr_14"]).all()
        # SL anchor must sit below the market for longs
        closes = df_m15.loc[fired.index, "close"]
        assert (fired["swing_price"] < closes).all()

    def test_no_regime_no_signal(self):
        flat = np.full(500, 1.10)  # no regime possible
        df_h1 = _make_h1(flat)
        df_m15, _ = _trending_market_with_pullback()
        frame = build_signal_frame(
            df_m15, df_h1,
            StrategyParams(use_adx_gate=False, entry_mode="pullback_rsi"),
        )
        assert (frame["signal"] == 0).all()

    def test_generate_candidate_returns_none_when_quiet(self):
        df_m15, df_h1 = _trending_market_with_pullback()
        # Kill the last bar's trigger by forcing RSI to stay high (no recross)
        candidate = generate_candidate(
            df_m15.iloc[:50], df_h1.iloc[:50],
            StrategyParams(use_adx_gate=False, entry_mode="pullback_rsi"),
        )
        assert candidate is None

    def test_generate_candidate_matches_last_frame_row(self):
        df_m15, df_h1 = _trending_market_with_pullback()
        params = StrategyParams(use_adx_gate=False, entry_mode="pullback_rsi")
        frame = build_signal_frame(df_m15, df_h1, params)
        trigger_times = frame.index[frame["signal"] == 1]
        assert len(trigger_times) > 0
        cut = df_m15.index.get_loc(trigger_times[-1]) + 1
        candidate = generate_candidate(df_m15.iloc[:cut], df_h1, params)
        assert isinstance(candidate, Candidate)
        assert candidate.direction == 1
        assert candidate.signal_time == trigger_times[-1]

    def test_model_never_creates_signals(self):
        """The strategy layer must be the only signal source."""
        import inspect

        source = inspect.getsource(strategy)
        assert "predict_signal" not in source
        assert "load_model" not in source


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestSpreadFilter:
    def test_spread_within_limit(self):
        assert spread_ok(0.8) is True
        assert spread_ok(config.MAX_SPREAD_PIPS) is True

    def test_spread_above_limit_blocks(self):
        assert spread_ok(config.MAX_SPREAD_PIPS + 0.01) is False


class TestVolatilityFilter:
    def test_atr_below_min_blocks(self):
        assert volatility_ok(config.ATR_MIN_PIPS - 0.5, 8.0) is False

    def test_atr_ok(self):
        assert volatility_ok(6.0, 6.0) is True

    def test_atr_spike_above_3x_median_blocks(self):
        assert volatility_ok(19.0, 6.0) is False  # 19 > 3×6

    def test_missing_median_only_checks_min(self):
        assert volatility_ok(6.0, None) is True
        assert volatility_ok(2.0, None) is False


class TestSessionFilter:
    # Europe/London is UTC+0 in winter, UTC+1 (BST) in summer.

    def test_winter_session_open_in_utc(self):
        # 08:30 London in January == 08:30 UTC
        assert entry_session_ok(datetime(2026, 1, 14, 8, 30, tzinfo=UTC)) is True
        # 07:30 UTC == 07:30 London → before session
        assert entry_session_ok(datetime(2026, 1, 14, 7, 30, tzinfo=UTC)) is False

    def test_summer_dst_shift(self):
        # 07:30 UTC in July == 08:30 BST → inside session (would fail with fixed UTC hours)
        assert entry_session_ok(datetime(2026, 7, 15, 7, 30, tzinfo=UTC)) is True
        # 16:30 UTC in July == 17:30 BST → after session close
        assert entry_session_ok(datetime(2026, 7, 15, 16, 30, tzinfo=UTC)) is False

    def test_session_close_boundary(self):
        # 17:00 London exactly → closed
        assert entry_session_ok(datetime(2026, 1, 14, 17, 0, tzinfo=UTC)) is False
        assert entry_session_ok(datetime(2026, 1, 14, 16, 59, tzinfo=UTC)) is True

    def test_friday_cutoff(self):
        # Friday 2026-01-16, 15:00 London → blocked; 14:59 → allowed
        assert entry_session_ok(datetime(2026, 1, 16, 15, 0, tzinfo=UTC)) is False
        assert entry_session_ok(datetime(2026, 1, 16, 14, 59, tzinfo=UTC)) is True

    def test_weekend_blocked(self):
        assert entry_session_ok(datetime(2026, 1, 17, 10, 0, tzinfo=UTC)) is False  # Sat
        assert entry_session_ok(datetime(2026, 1, 18, 10, 0, tzinfo=UTC)) is False  # Sun

    def test_sunday_open_block(self):
        # Sunday 22:30 London (30 min after open) → blocked
        assert entry_session_ok(datetime(2026, 1, 18, 22, 30, tzinfo=UTC)) is False

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError):
            entry_session_ok(datetime(2026, 1, 14, 9, 0))


# ---------------------------------------------------------------------------
# regime_daily entry mode (adopted from the 2026-07 ablation; research_log.md)
# ---------------------------------------------------------------------------

class TestRegimeDailyMode:
    def _frames(self):
        # Strong H1 uptrend + gently rising M15 series spanning several days
        h1_prices = np.linspace(1.05, 1.15, 500)
        df_h1 = _make_h1(h1_prices, start="2025-01-01")
        n = 960  # 10 days of M15 bars
        m15_prices = np.linspace(1.13, 1.15, n)
        start = df_h1.index[-1] - pd.Timedelta(minutes=15 * (n - 1))
        df_m15 = _make_m15(m15_prices, start=start)
        return df_m15, df_h1

    def test_at_most_one_signal_per_london_day(self):
        from zoneinfo import ZoneInfo
        df_m15, df_h1 = self._frames()
        params = StrategyParams(entry_mode="regime_daily", use_adx_gate=False)
        frame = build_signal_frame(df_m15, df_h1, params)
        fired = frame[frame["signal"] != 0]
        assert len(fired) > 0
        # entry executes at the NEXT bar; group those by London date
        entry_bars = [
            frame.index[frame.index.get_loc(t) + 1] for t in fired.index
            if frame.index.get_loc(t) + 1 < len(frame)
        ]
        days = pd.Series([t.tz_convert(ZoneInfo("Europe/London")).date() for t in entry_bars])
        assert days.value_counts().max() == 1

    def test_entry_bar_is_in_session_and_signal_bar_may_not_be(self):
        df_m15, df_h1 = self._frames()
        params = StrategyParams(entry_mode="regime_daily", use_adx_gate=False)
        frame = build_signal_frame(df_m15, df_h1, params)
        import strategy as strategy_module
        for t in frame.index[frame["signal"] != 0]:
            pos = frame.index.get_loc(t) + 1
            if pos < len(frame):
                assert strategy_module.entry_session_ok(frame.index[pos])

    def test_no_regime_no_daily_signal(self):
        df_m15, _ = self._frames()
        flat_h1 = _make_h1(np.full(500, 1.10))
        params = StrategyParams(entry_mode="regime_daily", use_adx_gate=False)
        frame = build_signal_frame(df_m15, flat_h1, params)
        assert (frame["signal"] == 0).all()

    def test_swing_is_nan_and_candidate_swing_none(self):
        df_m15, df_h1 = self._frames()
        params = StrategyParams(entry_mode="regime_daily", use_adx_gate=False)
        frame = build_signal_frame(df_m15, df_h1, params)
        fired = frame[frame["signal"] != 0]
        assert fired["swing_price"].isna().all()  # SL anchors at entry

        trigger_pos = frame.index.get_loc(fired.index[-1]) + 1
        candidate = generate_candidate(df_m15.iloc[:trigger_pos], df_h1, params)
        assert candidate is not None
        assert candidate.swing_price is None


class TestEntrySessionMask:
    def test_mask_matches_scalar_function(self):
        import strategy as strategy_module
        # spans winter, DST changeover, summer, Fridays and weekends
        index = pd.date_range("2026-03-25", "2026-04-08", freq="97min", tz="UTC")
        mask = strategy_module.entry_session_mask(index)
        for ts, ok in zip(index, mask):
            assert ok == strategy_module.entry_session_ok(ts), str(ts)
