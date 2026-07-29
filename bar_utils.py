"""
bar_utils.py

Finding 6 fix: exclude the still-forming (unclosed) trendbar from an OHLCV
frame. cTrader's GetTrendbars returns the in-progress period as the last row;
strategy.generate_candidate evaluates ``iloc[-1]`` expecting the last CLOSED
bar, so acting on live data would trade a partial candle.

Pure and dependency-light (pandas only) so it is trivially unit-testable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

UTC = timezone.utc


def drop_forming_bar(
    df: pd.DataFrame,
    period_minutes: int,
    now: datetime | None = None,
) -> pd.DataFrame:
    """
    Return ``df`` with any trailing bar whose period has NOT yet fully elapsed
    removed. A bar indexed at start time ``t`` is closed only once
    ``now >= t + period``; otherwise it is still forming and is dropped.

    Assumes a UTC DatetimeIndex of bar START times (as built by the adapters).
    Leaves closed bars untouched; a no-op when the last bar is already closed.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if now is None:
        now = datetime.now(UTC)
    now_ts = pd.Timestamp(now)
    if now_ts.tz is None:
        now_ts = now_ts.tz_localize("UTC")

    period = pd.Timedelta(minutes=period_minutes)
    # Keep only bars whose close time (start + period) is at or before now.
    closed = df.index + period <= now_ts
    if bool(closed.all()):
        return df
    return df.loc[closed]
