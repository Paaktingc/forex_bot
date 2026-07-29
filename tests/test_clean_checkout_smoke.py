"""
tests/test_clean_checkout_smoke.py

Clean-checkout integrity guard. This suite fails if the committed tree is not
self-contained — i.e. if committed `main.py`/startup depends on a module or
attribute that was never committed (the defect that motivated
`fix: restore self-contained deployment baseline`).

It imports the core modules and verifies startup reaches its retirement guard
WITHOUT any broker access.
"""
from __future__ import annotations

import importlib

import pytest

CORE_MODULES = [
    "config", "data_feed", "symbol_specs", "strategy", "risk_manager",
    "reconcile", "bar_utils", "trade_journal", "monte_carlo_dd",
    "execution", "news_filter", "main",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name):
    # A missing untracked dependency (e.g. reconcile.py) makes this raise
    # ModuleNotFoundError on a clean checkout.
    importlib.import_module(name)


def test_committed_main_dependencies_present():
    """The exact attributes committed main.py depends on must exist."""
    import risk_manager
    import trade_journal
    import reconcile
    import bar_utils

    assert hasattr(risk_manager.RiskManager, "load_or_init")
    assert hasattr(trade_journal, "get_open_positions")
    assert hasattr(reconcile, "reconcile_closures")
    assert hasattr(bar_utils, "drop_forming_bar")


def test_programme_constants_defined():
    """risk_manager references these programme constants; they must be defined
    so a clean checkout imports and initialises."""
    import config
    for const in ("OFFICIAL_DAILY_LOSS_PCT", "RISK_STATE_PATH",
                  "MIN_PROFITABLE_DAYS", "KILL_SWITCH_PCT", "RISK_PER_TRADE_PCT"):
        assert hasattr(config, const), f"config.{const} missing"


def test_startup_reaches_guard_without_broker(monkeypatch, tmp_path):
    """initialize_bot must reach the strategy-retirement guard BEFORE touching a
    broker. The repo default ENTRY_MODE is the retired H1 regime strategy, so a
    clean startup raises RetiredStrategyError and never calls connect_broker."""
    import main
    import strategy

    def _no_broker(*a, **k):
        raise AssertionError("startup must not access the broker in this test")

    monkeypatch.setattr(main.data_feed, "connect_broker", _no_broker)
    with pytest.raises(strategy.RetiredStrategyError):
        main.initialize_bot()
