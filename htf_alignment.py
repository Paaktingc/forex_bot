"""
htf_alignment.py — leak-free alignment of higher-timeframe features onto a
lower-timeframe decision index.

Why this exists
---------------
OHLCV bars in this repo are LEFT-labelled: a bar stamped ``o`` covers the
half-open interval ``[o, o + timeframe)`` and its close is not known until
``o + timeframe``. Any feature derived from that bar (EMA, ADX, swing, regime)
is therefore available only AT its close, i.e. at ``o + timeframe``, not at the
opening label ``o``.

Merging a higher-timeframe frame onto a lower-timeframe index with
``merge_asof(direction="backward")`` on the higher-frame LABEL matches a lower
bar to the higher bar CONTAINING it — which is still forming and whose close is
future information. That is the structural look-ahead leak documented in
``research_log.md`` (Gate 0, commit 9b49079).

Invariant enforced here
-----------------------
At a decision timestamp ``t`` (a lower-timeframe bar label), the only higher
bars that may be used are those already CLOSED:

    higher_bar_available_at <= t,   where  available_at = open + timeframe

and the selected row is the latest such row (or NaN if none has closed yet).

Left- vs right-labelled bars
----------------------------
This module assumes higher frames are LEFT-labelled (index = bar OPEN time),
which matches ``resampler.resample_ohlcv`` (``label="left"``) and the ``*_H1``
CSVs. If a frame is right-labelled (index = close time), pass
``already_close_labelled=True`` so no offset is added.
"""

from __future__ import annotations

import pandas as pd


def infer_timeframe(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most common positive spacing of a DatetimeIndex (the bar duration).

    Raises if the duration cannot be determined (fewer than two bars, or no
    positive gaps) — we never silently guess a timeframe.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("infer_timeframe: need a DatetimeIndex with >= 2 bars")
    diffs = index.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("infer_timeframe: no positive spacing between bars")
    return pd.Timedelta(diffs.mode().iloc[0])


def align_last_closed_bar(
    lower_index: pd.DatetimeIndex,
    higher_frame: pd.DataFrame,
    *,
    timeframe: pd.Timedelta | None = None,
    columns: list[str] | None = None,
    already_close_labelled: bool = False,
) -> pd.DataFrame:
    """Align ``higher_frame`` onto ``lower_index`` using closed-bar availability.

    For each lower-timeframe timestamp ``t`` in ``lower_index``, attach the row
    of ``higher_frame`` for the most recent higher bar that has CLOSED at or
    before ``t``. Rows for which no higher bar has closed yet are NaN.

    Parameters
    ----------
    lower_index : the lower-timeframe decision index (bar labels).
    higher_frame : higher-timeframe features, indexed by bar OPEN time
        (left-labelled) unless ``already_close_labelled`` is True.
    timeframe : higher-bar duration; inferred from ``higher_frame.index`` if None.
    columns : subset of columns to align (default: all).
    already_close_labelled : set True if ``higher_frame.index`` is close time.

    Safeguards: both sides are sorted; timezone-awareness must match on both
    sides (mixing naive and tz-aware raises); ``allow_exact_matches=True`` so a
    bar closing exactly at ``t`` is available at ``t``; a not-yet-closed bar can
    never be selected because availability uses the close timestamp.
    """
    if not isinstance(lower_index, pd.DatetimeIndex):
        lower_index = pd.DatetimeIndex(lower_index)
    if not isinstance(higher_frame.index, pd.DatetimeIndex):
        raise TypeError("align_last_closed_bar: higher_frame must have a DatetimeIndex")

    lower_tz = lower_index.tz is not None
    higher_tz = higher_frame.index.tz is not None
    if lower_tz != higher_tz:
        raise ValueError(
            "align_last_closed_bar: mixed tz-awareness — lower_index "
            f"tz-aware={lower_tz}, higher_frame tz-aware={higher_tz}"
        )

    cols = list(columns) if columns is not None else list(higher_frame.columns)

    right = higher_frame.loc[:, cols].sort_index().copy()
    if already_close_labelled:
        available_at = right.index
    else:
        tf = timeframe if timeframe is not None else infer_timeframe(right.index)
        available_at = right.index + tf
    right = right.assign(_available_at=available_at).sort_values("_available_at")

    left = pd.DataFrame(index=pd.DatetimeIndex(lower_index)).sort_index()

    merged = pd.merge_asof(
        left.reset_index(names="_t"),
        right.reset_index(names="_open").rename(columns={"_open": "_higher_open"}),
        left_on="_t",
        right_on="_available_at",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("_t")

    out = merged.loc[:, cols]
    # return in the caller's original index order
    out.index = left.index
    return out.reindex(pd.DatetimeIndex(lower_index))
