"""Cycle 6 portfolio-sleeve barrier model.

Reproduces the transparent synthetic scenarios from the Cycle 6 research
report.  This is a research/calibration tool, not a trading signal and not a
substitute for the empirical paired block-bootstrap required by the report.

Run with::

    python sleeve_model.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SEED = 7
TARGET_PCT = 6.0
KILL_PCT = 3.0
MAX_DAYS = 1_000


def stage_sim(generator, n_paths: int = 8_000) -> tuple[float, float, float, float]:
    """Return pass, kill, median-resolution-days, and unresolved rates."""
    daily_returns = generator(MAX_DAYS, n_paths).astype(np.float32)
    cumulative = np.cumsum(daily_returns, axis=1)
    hit_up = cumulative >= TARGET_PCT
    hit_down = cumulative <= -KILL_PCT
    t_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), MAX_DAYS + 1)
    t_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), MAX_DAYS + 1)
    pass_rate = float(np.mean(t_up < t_down))
    kill_rate = float(np.mean(t_down < t_up))
    unresolved = 1.0 - pass_rate - kill_rate
    median_days = float(np.median(np.minimum(t_up, t_down)))
    return pass_rate, kill_rate, median_days, unresolved


def london_daily(
    payoff: float,
    win_probability: float,
    risk_pct: float,
    frequency: float,
    rng: np.random.Generator,
):
    def generate(days: int, n_paths: int) -> np.ndarray:
        trades = rng.random((n_paths, days)) < frequency
        wins = rng.random((n_paths, days)) < win_probability
        return np.where(wins, payoff * risk_pct, -risk_pct) * trades

    return generate


def payoff_from_pf(profit_factor: float, win_probability: float) -> float:
    return profit_factor * (1.0 - win_probability) / win_probability


def combined_daily(
    london: tuple[float, float, float, float],
    sleeve: tuple[float, float, float, float],
    correlation: float,
    rng: np.random.Generator,
):
    """Construct correlated London/sleeve daily outcomes via Gaussian copula."""
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be between -1 and 1")
    b_l, p_l, risk_l, freq_l = london
    b_s, p_s, risk_s, freq_s = sleeve

    def generate(days: int, n_paths: int) -> np.ndarray:
        z = rng.standard_normal((n_paths, days, 2)).astype(np.float32)
        z_s = correlation * z[..., 0] + np.sqrt(1.0 - correlation**2) * z[..., 1]
        u_l, u_s = ndtr(z[..., 0]), ndtr(z_s)
        trade_l = rng.random((n_paths, days)) < freq_l
        trade_s = rng.random((n_paths, days)) < freq_s
        pnl_l = np.where(u_l < p_l, b_l * risk_l, -risk_l) * trade_l
        pnl_s = np.where(u_s < p_s, b_s * risk_s, -risk_s) * trade_s
        return pnl_l + pnl_s

    return generate


def run() -> None:
    rng = np.random.default_rng(SEED)
    pf_london = 1.165
    # Calibration selected in the report to reproduce the observed 66.4% pass.
    p_london, frequency_london = 0.35, 0.6
    b_london = payoff_from_pf(pf_london, p_london)

    print("Cycle 6 synthetic sleeve model (percent returns)")
    baseline = stage_sim(
        london_daily(b_london, p_london, 0.30, frequency_london, rng), 20_000
    )
    print(
        "London 0.30%: "
        f"Ppass={baseline[0]:.3f} Pkill={baseline[1]:.3f} "
        f"median_days={baseline[2]:.0f} P3={baseline[0] ** 3:.3f}"
    )

    print("\nRequired sleeve PF at rho=0, risk split 0.25%/0.25%")
    for sleeve_pf in (1.15, 1.25, 1.35, 1.45, 1.60, 1.80):
        p_sleeve = 0.45
        generator = combined_daily(
            (b_london, p_london, 0.25, frequency_london),
            (payoff_from_pf(sleeve_pf, p_sleeve), p_sleeve, 0.25, 0.8),
            0.0,
            rng,
        )
        p_pass, p_kill, days, _ = stage_sim(generator)
        print(
            f"PF {sleeve_pf:.2f}: Ppass={p_pass:.3f} Pkill={p_kill:.3f} "
            f"median_days={days:.0f} P3={p_pass**3:.3f}"
        )


if __name__ == "__main__":
    run()
