"""
tests/test_programme_risk_gate.py

The repo's hard survivability gate is P(maxDD > 5%) < 5% over a one-year
horizon, measured on the frozen return stream with NO kill switch and NO
absorption (see STEP 1 / GATE A, research_log.md, 2026-07-27). A programme
profile whose per-trade risk pushes P(maxDD>5%) at or above 5% is not
deployable regardless of its expected return.

This test fails if ANY programme profile in config.PROGRAMMES sets a
risk_per_trade_pct that breaches that gate on the current R distribution
(the frozen EURUSD ablation-(a) regime-only series). It is deliberately
grounded in the actual return stream, not a hard-coded risk ceiling, so it
keeps tracking the gate if the R series is re-frozen.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import config
from monte_carlo_dd import block_bootstrap_paths, DEFAULT_BLOCK_SIZE, RANDOM_SEED

_R_CSV = os.path.join(os.path.dirname(__file__), "..", "research",
                      "ablation_a_R_design.csv")
_MAXDD_LIMIT = 0.05          # 5% drawdown threshold
_GATE = 0.05                 # P(maxDD>5%) must be strictly below 5%
_TRADES_PER_YEAR = 111       # ablation (a) frequency on EURUSD; 1y maxDD horizon
_N_PATHS = 20_000


def _r_series() -> np.ndarray:
    r = pd.read_csv(_R_CSV)["r"].to_numpy(float)
    assert r.size > 500, f"R series looks truncated: n={r.size}"
    return r


def _p_maxdd_gt(r: np.ndarray, risk: float, threshold: float = _MAXDD_LIMIT,
                horizon: int = _TRADES_PER_YEAR) -> float:
    """P(max drawdown > threshold) over `horizon` trades on UNABSORBED paths.

    No kill switch, no absorption: this measures how risky the return stream
    itself is at the given per-trade risk fraction — the quantity the hard gate
    constrains.
    """
    paths = block_bootstrap_paths(r * risk, n_paths=_N_PATHS,
                                  block_size=DEFAULT_BLOCK_SIZE, seed=RANDOM_SEED)
    eq = np.cumprod(1.0 + paths[:, :horizon], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (peak - eq).max(axis=1) / peak.max(axis=1)
    return float((dd > threshold).mean())


@pytest.mark.parametrize("name", list(config.PROGRAMMES))
def test_programme_risk_within_maxdd_gate(name):
    r = _r_series()
    risk = config.PROGRAMMES[name]["risk_per_trade_pct"]
    p = _p_maxdd_gt(r, risk)
    assert p < _GATE, (
        f"programme {name!r} risk_per_trade_pct={risk:.4f} implies "
        f"P(maxDD>5%)={p:.4f} >= {_GATE} — over the hard survivability gate"
    )


def test_gate_actually_bites_on_old_defective_values():
    """Guardrail: the retired 0.30%/0.40% settings MUST fail the gate, so a
    silent regression back to them is caught. If this ever passes, the gate has
    stopped discriminating and the test above is worthless."""
    r = _r_series()
    assert _p_maxdd_gt(r, 0.003) >= _GATE   # was 15.3%
    assert _p_maxdd_gt(r, 0.004) >= _GATE   # was 35.2%
