"""
Causality / leakage tests for the intraday volatility-expansion continuation
study. The whole point of this research line is that its (negative) result is
LEAK-FREE, so these are mandatory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.fx_intraday_volatility_continuation import (
    config as C, prepare_data as P, signal as S, backtest as B,
)


# --- data / split integrity -------------------------------------------------

@pytest.mark.parametrize("pair", C.PAIRS)
def test_data_left_labelled_utc_monotonic(pair):
    df = P.load_m15(pair).head(2000)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df.index.duplicated().sum() == 0
    # left-labelled: 4 consecutive M15 opens compose one hour; first open is the
    # hour's open, spacing is 15m
    assert (df.index[1] - df.index[0]) == pd.Timedelta(minutes=15)


def test_lockbox_guard_blocks_and_unlocks():
    with pytest.raises(P.LockboxViolation):
        P.load_split("EURUSD", "locked_oos")
    df = P.load_split("EURUSD", "locked_oos", unlock_oos=True)
    assert df.index.min() >= pd.Timestamp("2023-01-01", tz="UTC")


def test_session_flag_is_dst_aware():
    # 12:00 UTC in summer is 13:00 London (in session); in winter 12:00 UTC is
    # 12:00 London (in session). 16:30 UTC summer = 17:30 London (out).
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-07-01 07:30", tz="UTC"),   # 08:30 London -> in
        pd.Timestamp("2020-07-01 15:30", tz="UTC"),   # 16:30 London -> out (close 16:45>16)
        pd.Timestamp("2020-01-01 07:45", tz="UTC"),   # 07:45 London close 08:00 -> in
    ])
    df = pd.DataFrame({"open": 1., "high": 1., "low": 1., "close": 1.}, index=idx)
    flag = P.add_session_flag(df).to_numpy()
    assert flag[0] and (not flag[1]) and flag[2]


# --- causality of the signal ------------------------------------------------

def _sample_df(pair="EURUSD", n=6000):
    return P.load_split(pair, "discovery").head(n)


def test_atr_reference_excludes_current_bar():
    df = _sample_df().copy()
    atr = S.atr_reference(df)
    t = 500
    # perturb bar t's OHLC massively; atr_ref[t] must NOT change (it is ATR
    # through t-1), while atr_ref[t+1] MAY change.
    df2 = df.copy()
    df2.iloc[t, df2.columns.get_loc("high")] *= 5
    df2.iloc[t, df2.columns.get_loc("low")] *= 0.2
    atr2 = S.atr_reference(df2)
    assert atr.iloc[t] == pytest.approx(atr2.iloc[t], nan_ok=True)
    assert not np.isclose(atr.iloc[t + 1], atr2.iloc[t + 1])


def test_signal_is_truncation_invariant():
    # The direction at bar t must be identical whether computed on the full
    # history or on data truncated at t (no future information used).
    df = _sample_df()
    full = S.raw_signal_frame(df)["direction"]
    sig_bars = full[full != 0].index[:15]
    for ts in sig_bars:
        i = df.index.get_loc(ts)
        trunc = S.raw_signal_frame(df.iloc[: i + 1])["direction"]
        assert trunc.iloc[-1] == full.loc[ts], f"future leak at {ts}"


def test_breakout_excludes_current_bar():
    df = _sample_df()
    frame = S.raw_signal_frame(df)
    long_bars = frame.index[frame["direction"] == 1][:20]
    for ts in long_bars:
        i = df.index.get_loc(ts)
        prev8_high = df["high"].iloc[i - C.BREAKOUT_LOOKBACK:i].max()
        assert df["close"].iloc[i] > prev8_high     # broke prior-8, not incl self


def test_entry_is_next_bar_open():
    df = _sample_df()
    tr = B.run(df, "EURUSD")
    assert len(tr) > 0
    d = (pd.to_datetime(tr["entry_time"]) - pd.to_datetime(tr["signal_time"]))
    assert (d == pd.Timedelta(minutes=15)).all()    # entry strictly next bar


def test_usdjpy_pip_scaling_in_costs():
    # A USDJPY trade's commission-in-R must use pip_size=0.01, not 0.0001.
    from symbol_specs import get_symbol_spec
    assert get_symbol_spec("USDJPY").pip_size == 0.01
    df = P.load_split("USDJPY", "discovery").head(8000)
    tr = B.run(df, "USDJPY")
    assert len(tr) > 0 and np.isfinite(tr["r"]).all()


# --- static guards ----------------------------------------------------------

def test_no_negative_shift_or_centered_windows():
    import inspect
    from research.fx_intraday_volatility_continuation import signal, backtest, baselines
    for mod in (signal, backtest, baselines):
        src = inspect.getsource(mod)
        assert ".shift(-" not in src, f"negative shift in {mod.__name__}"
        assert "center=True" not in src, f"centred window in {mod.__name__}"
