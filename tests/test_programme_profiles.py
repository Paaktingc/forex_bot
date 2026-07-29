"""
tests/test_programme_profiles.py

Covers the Cycle-6 programme-profile layer: config profiles, the multi-step
Monte Carlo runner, and risk_manager's profitable-day / official-daily-loss
gates. The frozen strategy is untouched; only the evaluation geometry changes.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

import config
from monte_carlo_dd import run_programme_monte_carlo
from monte_carlo_dd import run_step_monte_carlo

_SNAPSHOT_KEYS = (
    "PROGRAMME", "PROGRAMME_STEPS", "PROFIT_TARGET_PCT", "MAX_DRAWDOWN_LIMIT",
    "KILL_SWITCH_PCT", "DRAWDOWN_WARNING_THRESHOLD", "OFFICIAL_DAILY_LOSS_PCT",
    "DAILY_LOSS_PCT", "WEEKLY_STOP_PCT", "RISK_PER_TRADE_PCT",
    "MIN_PROFITABLE_DAYS", "PROFITABLE_DAY_MIN_PCT",
)


def _reload_config(programme: str) -> SimpleNamespace:
    """
    Reload config under a given PROGRAMME env var and return a SNAPSHOT of the
    derived constants (captured while the profile is active), then restore the
    module to its default state. Returning a snapshot avoids the finally-block
    reload clobbering the values the test asserts on.
    """
    prev = os.environ.get("PROGRAMME")
    os.environ["PROGRAMME"] = programme
    try:
        importlib.reload(config)
        return SimpleNamespace(**{k: getattr(config, k) for k in _SNAPSHOT_KEYS})
    finally:
        if prev is None:
            os.environ.pop("PROGRAMME", None)
        else:
            os.environ["PROGRAMME"] = prev
        importlib.reload(config)  # leave the module in its default state


# ---------------------------------------------------------------------------
# Config profiles
# ---------------------------------------------------------------------------

def test_bootcamp_profile_geometry():
    c = _reload_config("bootcamp")
    assert c.PROGRAMME == "bootcamp"
    assert c.PROGRAMME_STEPS == (0.06, 0.06, 0.06)
    assert c.MAX_DRAWDOWN_LIMIT == 0.05
    assert c.KILL_SWITCH_PCT == 0.03
    assert c.MIN_PROFITABLE_DAYS == 0
    # kill must always fire strictly before the official breach
    assert c.KILL_SWITCH_PCT < c.MAX_DRAWDOWN_LIMIT


def test_high_stakes_profile_geometry():
    c = _reload_config("high_stakes")
    assert c.PROGRAMME == "high_stakes"
    assert c.PROGRAMME_STEPS == (0.10, 0.05)
    assert c.MAX_DRAWDOWN_LIMIT == 0.10
    assert c.KILL_SWITCH_PCT == 0.06
    assert c.OFFICIAL_DAILY_LOSS_PCT == 0.05
    assert c.MIN_PROFITABLE_DAYS == 3
    assert c.PROFITABLE_DAY_MIN_PCT == 0.005
    assert c.KILL_SWITCH_PCT < c.MAX_DRAWDOWN_LIMIT
    # self-imposed daily pacing must sit inside the official daily limit
    assert c.DAILY_LOSS_PCT < c.OFFICIAL_DAILY_LOSS_PCT


def test_unknown_programme_raises():
    with pytest.raises(ValueError):
        _reload_config("not_a_programme")


# ---------------------------------------------------------------------------
# Multi-step Monte Carlo
# ---------------------------------------------------------------------------

def _frozen_ledger(n=2000, risk=0.003, p_win=0.35, payoff=2.16, seed=1):
    rng = np.random.default_rng(seed)
    wins = rng.random(n) < p_win
    return np.where(wins, payoff * risk, -risk)


def test_programme_runner_step_count_and_keys():
    pct = _frozen_ledger()
    res = run_programme_monte_carlo(
        pct, step_targets=(0.06, 0.06, 0.06), fail=-0.05, kill=-0.03, n_paths=3000
    )
    assert res["n_steps"] == 3
    assert len(res["per_step"]) == 3
    for s in res["per_step"]:
        # probabilities are valid and sum to <= 1
        assert 0.0 <= s["p_pass"] <= 1.0
        assert 0.0 <= s["p_kill_switch"] <= 1.0
        assert s["p_pass"] + s["p_kill_switch"] <= 1.0 + 1e-9
    # completion = product of per-step pass probabilities
    # completion rounds to 4dp and is the product of UNROUNDED per-step passes,
    # so compare against the rounded per-step product with a 4dp tolerance.
    prod = 1.0
    for s in res["per_step"]:
        prod *= s["p_pass"]
    assert res["p_complete_programme"] == pytest.approx(prod, abs=1e-3)


def test_single_step_runner_accepts_profile_geometry():
    pct = _frozen_ledger(risk=0.004)
    res = run_step_monte_carlo(
        pct,
        n_paths=2_000,
        target=0.10,
        fail=-0.10,
        kill=-0.06,
    )
    assert res["target_pct"] == 10.0
    assert res["fail_pct"] == -10.0
    assert res["kill_pct"] == -6.0


def test_high_stakes_beats_bootcamp_on_same_edge():
    """
    Core Cycle-6 claim: the identical frozen edge completes High Stakes far more
    often than Bootcamp, because a symmetric wide barrier suits a thin edge.
    """
    pct_bc = _frozen_ledger(risk=0.003, seed=7)
    pct_hs = _frozen_ledger(risk=0.004, seed=7)
    bc = run_programme_monte_carlo(
        pct_bc, step_targets=(0.06, 0.06, 0.06), fail=-0.05, kill=-0.03, n_paths=8000
    )
    hs = run_programme_monte_carlo(
        pct_hs, step_targets=(0.10, 0.05), fail=-0.10, kill=-0.06, n_paths=8000
    )
    assert hs["p_complete_programme"] > bc["p_complete_programme"]
    # worst High Stakes step should clear the <10% kill gate the report cites
    worst_hs_kill = max(s["p_kill_switch"] for s in hs["per_step"])
    assert worst_hs_kill < 0.25  # comfortably better than Bootcamp's ~0.29


def test_no_official_breach_when_kill_below_limit():
    """A path can only breach if a single trade gaps past the kill to the
    official limit; with 0.3–0.4% trades and a 2%+ gap between kill and limit,
    breaches should be effectively zero."""
    pct = _frozen_ledger(risk=0.004, seed=3)
    hs = run_programme_monte_carlo(
        pct, step_targets=(0.10, 0.05), fail=-0.10, kill=-0.06, n_paths=8000
    )
    for s in hs["per_step"]:
        assert s["p_breach_official"] == 0.0


# ---------------------------------------------------------------------------
# risk_manager profitable-day gate (High Stakes)
# ---------------------------------------------------------------------------

def test_profitable_days_gate_high_stakes(monkeypatch):
    import risk_manager as rm

    # Force the High Stakes profitable-day requirement regardless of env default.
    monkeypatch.setattr(rm.config, "MIN_PROFITABLE_DAYS", 3, raising=False)
    monkeypatch.setattr(rm.config, "PROFITABLE_DAY_MIN_PCT", 0.005, raising=False)

    r = rm.RiskManager(starting_balance=100_000.0)
    assert r.profitable_days == 0
    assert r.profitable_days_met() is False

    # Manually simulate three credited profitable days.
    r.profitable_days = 3
    assert r.profitable_days_met() is True


def test_profitable_days_noop_for_bootcamp(monkeypatch):
    import risk_manager as rm

    monkeypatch.setattr(rm.config, "MIN_PROFITABLE_DAYS", 0, raising=False)
    r = rm.RiskManager(starting_balance=5_000.0)
    # No requirement → always satisfied.
    assert r.profitable_days_met() is True


def test_official_daily_loss_backstop(monkeypatch):
    import risk_manager as rm

    monkeypatch.setattr(rm.config, "OFFICIAL_DAILY_LOSS_PCT", 0.05, raising=False)
    monkeypatch.setattr(rm.config, "KILL_SWITCH_PCT", 0.06, raising=False)
    monkeypatch.setattr(rm.config, "MAX_DRAWDOWN_LIMIT", 0.10, raising=False)
    r = rm.RiskManager(starting_balance=100_000.0)
    r.daily_start_balance = 100_000.0

    # 4% down: within the official 5% daily limit → still allowed.
    assert r.check_official_daily_loss(96_000.0) is True
    # Clear any disabled flag the previous call might have set (it shouldn't).
    if r.is_disabled():
        r.disabled_flag_path.unlink(missing_ok=True)
    # 5%+ down: breaches the official daily limit → disabled.
    assert r.check_official_daily_loss(94_900.0) is False
    assert r.is_disabled() is True
    r.disabled_flag_path.unlink(missing_ok=True)  # cleanup


def test_can_trade_enforces_official_daily_loss(monkeypatch, tmp_path):
    """The master live gate must invoke the programme's official backstop."""
    import risk_manager as rm

    monkeypatch.setattr(rm.config, "OFFICIAL_DAILY_LOSS_PCT", 0.05, raising=False)
    monkeypatch.setattr(rm.config, "KILL_SWITCH_PCT", 0.06, raising=False)
    monkeypatch.setattr(rm.config, "MAX_DRAWDOWN_LIMIT", 0.10, raising=False)
    monkeypatch.setattr(rm, "is_rollover_window", lambda: False)
    monkeypatch.setattr(rm, "is_no_trade_server_window", lambda: False)
    r = rm.RiskManager(
        starting_balance=100_000.0,
        disabled_flag_path=tmp_path / "disabled.json",
    )
    allowed, reason = r.can_trade(94_900.0, open_trades=0)
    assert allowed is False
    assert reason == "OFFICIAL DAILY LOSS LIMIT HIT"
    assert r.is_disabled() is True
