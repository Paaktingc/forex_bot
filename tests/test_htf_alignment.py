"""
tests/test_htf_alignment.py

Leakage-focused suite for htf_alignment.align_last_closed_bar and its use in
strategy.build_signal_frame. Every test encodes the invariant:

    an H1 (higher-timeframe) feature may be used at a decision timestamp t only
    if the H1 bar's CLOSE timestamp (open + timeframe) is <= t.

The synthetic cases are built so a still-forming bar carries an OBVIOUSLY
different value from the last closed bar, making look-ahead impossible to miss.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from htf_alignment import align_last_closed_bar, infer_timeframe


def _h1(labels_utc, values):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in labels_utc])
    return pd.DataFrame({"regime": values}, index=idx)


def _m15_index(start, n, tz="UTC"):
    return pd.date_range(pd.Timestamp(start, tz=tz), periods=n, freq="15min")


# ---------------------------------------------------------------------------
# 1. H1 -> M15 last-closed alignment + 4/5/6. intra-hour, boundary, first-close
# ---------------------------------------------------------------------------

def test_intrahour_decisions_do_not_see_forming_bar():
    # 08:00 bar = bearish(-1); 09:00 bar = bullish(+1). Decisions inside the
    # 09:00 interval must remain bearish (the 09:00 bar closes at 10:00).
    h1 = _h1(["2020-06-01 07:00", "2020-06-01 08:00", "2020-06-01 09:00"],
             [0, -1, +1])
    m15 = _m15_index("2020-06-01 09:00", 4)  # 09:00,09:15,09:30,09:45
    out = align_last_closed_bar(m15, h1)["regime"]
    assert list(out) == [-1, -1, -1, -1], list(out)


def test_exact_close_boundary_makes_bar_available():
    h1 = _h1(["2020-06-01 08:00", "2020-06-01 09:00"], [-1, +1])
    # At 10:00 the 09:00 bar has closed and becomes usable.
    out = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2020-06-01 10:00", tz="UTC")]), h1)["regime"]
    assert out.iloc[0] == +1
    # At 09:59 it has not.
    out2 = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2020-06-01 09:59", tz="UTC")]), h1)["regime"]
    assert out2.iloc[0] == -1


def test_no_feature_before_first_h1_close():
    h1 = _h1(["2020-06-01 08:00", "2020-06-01 09:00"], [-1, +1])
    # Decisions before 09:00 (first close) have no closed H1 bar -> NaN.
    m15 = _m15_index("2020-06-01 08:00", 4)  # 08:00..08:45, first close at 09:00
    out = align_last_closed_bar(m15, h1)["regime"]
    assert out.isna().all(), list(out)


# ---------------------------------------------------------------------------
# 6. missing higher-timeframe bars
# ---------------------------------------------------------------------------

def test_missing_h1_bar_falls_back_to_earlier_closed_bar():
    # 09:00 bar is ABSENT (gap). A decision at 10:30 must fall back to the
    # 08:00 bar (closed 09:00), never forward to a later bar.
    h1 = _h1(["2020-06-01 08:00", "2020-06-01 10:00"], [-1, +1])
    out = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2020-06-01 10:30", tz="UTC")]), h1)["regime"]
    assert out.iloc[0] == -1  # 10:00 bar not closed until 11:00


# ---------------------------------------------------------------------------
# 7/8. timezone-aware + DST transition
# ---------------------------------------------------------------------------

def test_timezone_aware_alignment_and_mixed_tz_rejected():
    h1 = _h1(["2020-06-01 08:00", "2020-06-01 09:00"], [-1, +1])
    naive = pd.DatetimeIndex([pd.Timestamp("2020-06-01 10:00")])
    with pytest.raises(ValueError, match="mixed tz"):
        align_last_closed_bar(naive, h1)


def test_dst_transition_uses_utc_availability():
    # UK spring-forward 2021-03-28 01:00 UTC. Availability is computed in UTC,
    # so the closed-bar logic is unaffected by the wall-clock jump.
    h1 = _h1(["2021-03-28 00:00", "2021-03-28 01:00"], [-1, +1])
    # 01:15 UTC: only the 00:00 bar (closed 01:00) is available.
    out = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2021-03-28 01:15", tz="UTC")]), h1)["regime"]
    assert out.iloc[0] == -1
    # 02:00 UTC: the 01:00 bar has closed.
    out2 = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2021-03-28 02:00", tz="UTC")]), h1)["regime"]
    assert out2.iloc[0] == +1


# ---------------------------------------------------------------------------
# timeframe inference + safeguards
# ---------------------------------------------------------------------------

def test_infer_timeframe_h1_and_h4():
    h1 = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    h4 = pd.date_range("2020-01-01", periods=10, freq="4h", tz="UTC")
    assert infer_timeframe(h1) == pd.Timedelta(hours=1)
    assert infer_timeframe(h4) == pd.Timedelta(hours=4)


def test_h4_alignment_respects_4h_close():
    # H4 bars: 08:00 (-1) closes 12:00; 12:00 (+1) closes 16:00.
    h4 = pd.DataFrame(
        {"regime": [-1, +1]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-06-01 08:00", tz="UTC"),
                                pd.Timestamp("2020-06-01 12:00", tz="UTC")]),
    )
    # 11:00 -> only 08:00 bar closed? No: 08:00 closes 12:00. Nothing closed yet.
    out = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2020-06-01 11:00", tz="UTC")]), h4,
                                timeframe=pd.Timedelta(hours=4))["regime"]
    assert pd.isna(out.iloc[0])
    # 13:00 -> 08:00 bar closed at 12:00.
    out2 = align_last_closed_bar(pd.DatetimeIndex([pd.Timestamp("2020-06-01 13:00", tz="UTC")]), h4,
                                 timeframe=pd.Timedelta(hours=4))["regime"]
    assert out2.iloc[0] == -1


def test_infer_timeframe_needs_two_bars():
    one = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")])
    with pytest.raises(ValueError):
        infer_timeframe(one)


# ---------------------------------------------------------------------------
# 9/10/11. rolling indicators are trailing, no centred windows, no neg shift
# ---------------------------------------------------------------------------

def test_features_use_trailing_windows_only():
    import inspect
    import features
    src = inspect.getsource(features)
    assert "center=True" not in src, "centred rolling window in features.py"
    assert ".shift(-" not in src, "negative shift (future data) in features.py"


def test_strategy_no_negative_shift():
    import inspect
    import strategy
    src = inspect.getsource(strategy)
    assert ".shift(-" not in src, "negative shift (future data) in strategy.py"
    assert "center=True" not in src, "centred rolling window in strategy.py"


# ---------------------------------------------------------------------------
# 2nd confirmed leak: features.add_h1_trend must also use last-closed alignment
# ---------------------------------------------------------------------------

def test_add_h1_trend_does_not_use_forming_bar():
    from features import add_h1_trend

    # H1: 07:00 & 08:00 bars are clearly bearish (close << EMA proxy); the
    # 09:00 bar is clearly bullish. Build EMA-friendly series so the 09:00
    # close is above its EMA and the earlier closes are below theirs.
    h1_idx = pd.date_range("2020-06-01 00:00", periods=12, freq="1h", tz="UTC")
    close = np.concatenate([np.linspace(1.20, 1.10, 9), [1.30, 1.31, 1.32]])
    h1 = pd.DataFrame({"open": close, "high": close + 1e-3,
                       "low": close - 1e-3, "close": close}, index=h1_idx)

    m15_idx = pd.date_range("2020-06-01 09:00", periods=4, freq="15min", tz="UTC")
    m15 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                        "volume": 0.0}, index=m15_idx)

    out = add_h1_trend(m15, h1)
    # The 09:00 H1 bar (bullish) closes at 10:00; decisions in [09:00,10:00)
    # must reflect the 08:00 bar (bearish), i.e. h1_trend == -1.
    assert (out["h1_trend"] == -1).all(), out["h1_trend"].tolist()
