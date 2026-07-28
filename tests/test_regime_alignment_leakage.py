"""
tests/test_regime_alignment_leakage.py

GATE 0 (2026-07-28) look-ahead finding, now FIXED. The H1 CSVs are LEFT-labelled:
a bar stamped `o` covers [o, o+1h) and only closes at o+1h. The old
strategy.build_signal_frame aligned the H1 regime onto M15 with
merge_asof(direction="backward") on the H1 LABEL, so an M15 bar at :00/:15/:30
was matched to the H1 bar CONTAINING it — still forming, its close future
information. The fix (htf_alignment.align_last_closed_bar) aligns on closed-bar
availability instead.

Invariant that must hold: the H1 regime aligned onto an M15 bar at time t must
come from an H1 bar that has CLOSED at or before t. These are now MANDATORY
passing tests — they fail if the leaky label-based alignment is ever reintroduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import strategy as S


def _eurusd_slice(n_days: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    d15 = backtest.get_ohlcv_from_csv("EURUSD", "M15")
    dh1 = backtest.get_ohlcv_from_csv("EURUSD", "H1")
    lo = pd.Timestamp("2015-01-01", tz="UTC")
    hi = lo + pd.Timedelta(days=n_days)
    return d15[(d15.index >= lo) & (d15.index < hi)], dh1[(dh1.index >= lo) & (dh1.index < hi)]


def test_h1_bars_are_left_labelled():
    """Confirms the labelling convention the leak depends on: an H1 bar's close
    equals the last M15 close inside [label, label+1h), i.e. it closes at
    label+1h, not at `label`."""
    d15, dh1 = _eurusd_slice(3)
    o = dh1.index[5]
    m15_in = d15[(d15.index >= o) & (d15.index < o + pd.Timedelta(hours=1))]
    assert len(m15_in) == 4
    assert m15_in["open"].iloc[0] == pytest.approx(dh1.loc[o, "open"])
    assert m15_in["close"].iloc[-1] == pytest.approx(dh1.loc[o, "close"])


def test_regime_alignment_uses_only_closed_h1_bars():
    d15, dh1 = _eurusd_slice(60)
    frame = S.build_signal_frame(d15, dh1)
    reg_m15 = frame["regime"]

    reg_h1 = S.h1_regime(dh1)
    # Independent reference: for each M15 bar, the regime of the latest H1 bar
    # whose CLOSE (label + 1h) is <= the M15 timestamp. Built without the
    # production utility so this cross-checks it rather than dogfooding it.
    close_frame = pd.DataFrame(
        {"regime": reg_h1.to_numpy(), "avail": reg_h1.index + pd.Timedelta(hours=1)}
    ).sort_values("avail")
    left = pd.DataFrame({"t": reg_m15.index})
    honest = pd.merge_asof(
        left, close_frame, left_on="t", right_on="avail",
        direction="backward", allow_exact_matches=True,
    )["regime"].fillna(0).astype(int).to_numpy()

    # No M15 bar may carry a regime that differs from the last CLOSED H1 bar's
    # regime (a difference means it is reading a not-yet-closed bar).
    mism = int((reg_m15.to_numpy() != honest).sum())
    assert mism == 0, f"{mism} M15 bars align to a not-yet-closed H1 bar (look-ahead)"
