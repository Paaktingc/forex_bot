"""
tests/test_regime_alignment_leakage.py

GATE 0 (2026-07-28) look-ahead finding. The H1 CSVs are LEFT-labelled: a bar
stamped `o` covers [o, o+1h) and only closes at o+1h. strategy.build_signal_frame
aligns the H1 regime onto M15 with merge_asof(direction="backward") on the H1
LABEL, so an M15 bar at :00/:15/:30 is matched to the H1 bar CONTAINING it —
which has not closed yet. The regime of that bar depends on its own (future)
close, so the aligned regime carries look-ahead.

Invariant that must hold for a leak-free signal: the H1 regime aligned onto an
M15 bar at time t must come from an H1 bar that has CLOSED at or before t.

This test encodes that invariant against the production alignment. It is
xfail(strict=True): it documents the known leak and will start PASSING the
moment the alignment is fixed (align on H1 close time = label+1h, or use the
last fully-closed H1 bar), at which point the marker should be removed.
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


@pytest.mark.xfail(strict=True, reason="GATE 0 leak: build_signal_frame aligns "
                   "regime on the H1 label, matching M15 bars to their still-"
                   "forming H1 bar. Fix: align on H1 close time. See research_log.md.")
def test_regime_alignment_uses_only_closed_h1_bars():
    d15, dh1 = _eurusd_slice(60)
    frame = S.build_signal_frame(d15, dh1)
    reg_m15 = frame["regime"]

    reg_h1 = S.h1_regime(dh1)
    # close time of each H1 bar (left-labelled -> closes at label + 1h)
    close_time = reg_h1.index + pd.Timedelta(hours=1)
    avail = pd.Series(reg_h1.to_numpy(), index=close_time)
    honest = pd.merge_asof(
        pd.DataFrame(index=reg_m15.index).reset_index(names="t"),
        avail.rename("regime").reset_index(names="t").sort_values("t"),
        on="t", direction="backward",
    ).set_index("t")["regime"].fillna(0).astype(int)

    # No M15 bar may carry a regime that differs from the last CLOSED H1 bar's
    # regime (a difference means it is reading a not-yet-closed bar).
    mism = int((reg_m15.to_numpy() != honest.to_numpy()).sum())
    assert mism == 0, f"{mism} M15 bars align to a not-yet-closed H1 bar (look-ahead)"
