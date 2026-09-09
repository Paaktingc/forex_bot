"""
tests/test_strategy_retirement.py

The H1 regime strategy (entry_mode regime_daily / regime_daily2) is retired for
structural higher-timeframe look-ahead leakage. These tests confirm it cannot be
activated for live/demo trading, while leaving research paths untouched.
"""

from __future__ import annotations

import pytest

import config
import strategy


@pytest.mark.parametrize("mode", ["regime_daily", "regime_daily2"])
def test_retired_modes_rejected(mode):
    with pytest.raises(strategy.RetiredStrategyError):
        strategy.assert_live_entry_mode_enabled(mode)


@pytest.mark.parametrize("mode", ["pullback_rsi", "range_fade"])
def test_non_retired_modes_allowed(mode):
    # Must not raise (these are not the retired H1 regime strategy).
    strategy.assert_live_entry_mode_enabled(mode)


def test_default_config_entry_mode_is_retired():
    # The repo default ENTRY_MODE is the retired strategy; the guard is what
    # keeps that config from silently trading.
    assert config.ENTRY_MODE in strategy.RETIRED_ENTRY_MODES


def test_initialize_bot_refuses_before_touching_broker(monkeypatch):
    import main

    def _boom():
        raise AssertionError("connect_broker must NOT be called for a retired mode")

    monkeypatch.setattr(main.data_feed, "connect_broker", _boom)
    monkeypatch.setattr(config, "ENTRY_MODE", "regime_daily", raising=False)
    with pytest.raises(strategy.RetiredStrategyError):
        main.initialize_bot()
