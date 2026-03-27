"""
trade_journal.py

CSV-backed trade journal and lightweight performance dashboard helpers.
All journal timestamps and daily rollups are handled in UTC to stay aligned
with the rest of the trading bot.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

JOURNAL_FILE = config.JOURNAL_PATH
JOURNAL_COLUMNS = [
    "ticket",
    "timestamp",
    "symbol",
    "direction",
    "entry_price",
    "sl",
    "tp",
    "lot_size",
    "confidence",
    "drawdown_at_entry",
    "session",
    "exit_price",
    "exit_time",
    "pnl_pips",
    "pnl_currency",
    "result",
    "exit_reason",
]

_COLUMN_ALIASES = {
    "action": "direction",
    "lots": "lot_size",
    "entry_time": "timestamp",
    "close_price": "exit_price",
    "close_time": "exit_time",
    "profit": "pnl_currency",
}


def _journal_path() -> Path:
    return Path(JOURNAL_FILE)


def _timestamp_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _format_timestamp(value: datetime | pd.Timestamp | None = None) -> str:
    if value is None:
        value = _timestamp_now()
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat()


def _parse_timestamp(value: Any, default: datetime | None = None) -> datetime:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default or _timestamp_now()

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return default or _timestamp_now()
    return parsed.to_pydatetime()


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return not pd.isna(value)
    except TypeError:
        return True


def _infer_session(timestamp: datetime) -> str:
    hour = timestamp.astimezone(UTC).hour
    if 13 <= hour <= 16:
        return "OVERLAP"
    if config.LONDON_START_UTC <= hour <= config.LONDON_END_UTC:
        return "LONDON"
    if config.NY_START_UTC <= hour <= config.NY_END_UTC:
        return "NEW_YORK"
    return "ASIA"


def _pip_size(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def _pip_value_per_lot(symbol: str) -> float:
    # The bot currently trades EURUSD by default, so the standard
    # USD-quoted major pair pip value is the project-consistent baseline.
    return 10.0


def _pnl_pips(entry_price: float, exit_price: float, direction: str, symbol: str) -> float:
    move = exit_price - entry_price
    if direction.upper() == "SELL":
        move = -move
    return round(move / _pip_size(symbol), 1)


def _pnl_currency(pnl_pips: float, lot_size: float, symbol: str) -> float:
    return round(pnl_pips * lot_size * _pip_value_per_lot(symbol), 2)


def _risk_pips(row: pd.Series) -> float | None:
    entry_price = _coerce_float(row.get("entry_price"))
    sl = _coerce_float(row.get("sl"))
    symbol = str(row.get("symbol") or config.SYMBOL)
    if entry_price is None or sl is None:
        return None
    risk = abs(entry_price - sl) / _pip_size(symbol)
    return risk if risk > 0 else None


def _rr_multiple(row: pd.Series) -> float | None:
    pnl_pips = _coerce_float(row.get("pnl_pips"))
    risk_pips = _risk_pips(row)
    if pnl_pips is None or risk_pips in (None, 0):
        return None
    return pnl_pips / risk_pips


def _result_from_pnl(pnl_currency: float) -> str:
    rounded = round(pnl_currency, 2)
    if rounded > 0:
        return "WIN"
    if rounded < 0:
        return "LOSS"
    return "BE"


def _ensure_journal_exists() -> None:
    journal_path = _journal_path()
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if not journal_path.exists():
        pd.DataFrame(columns=JOURNAL_COLUMNS).to_csv(journal_path, index=False)


def _normalize_existing_journal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for old_name, new_name in _COLUMN_ALIASES.items():
        if new_name not in out.columns and old_name in out.columns:
            out[new_name] = out[old_name]

    if "result" not in out.columns and "status" in out.columns:
        out["result"] = ""
        pnl_series = pd.to_numeric(out.get("pnl_currency"), errors="coerce")
        closed_mask = out["status"].astype(str).str.upper().eq("CLOSED")
        out.loc[closed_mask, "result"] = pnl_series.loc[closed_mask].fillna(0.0).apply(_result_from_pnl)

    if "direction" in out.columns:
        out["direction"] = out["direction"].astype(str).str.upper()
    if "result" in out.columns:
        out["result"] = out["result"].astype(str).replace({"<NA>": ""}).str.upper()
    if "exit_reason" in out.columns:
        out["exit_reason"] = out["exit_reason"].astype(str).replace({"<NA>": ""}).str.upper()
    if "session" in out.columns:
        out["session"] = out["session"].astype(str).replace({"<NA>": ""}).str.upper()
    if "session" not in out.columns and "timestamp" in out.columns:
        timestamps = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["session"] = timestamps.apply(
            lambda ts: _infer_session(ts.to_pydatetime()) if not pd.isna(ts) else ""
        )

    for column in JOURNAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    return out[JOURNAL_COLUMNS]


def _read_journal() -> pd.DataFrame:
    _ensure_journal_exists()
    df = pd.read_csv(_journal_path())
    return _normalize_existing_journal(df)


def _write_journal(df: pd.DataFrame) -> None:
    normalized = _normalize_existing_journal(df)
    normalized.to_csv(_journal_path(), index=False)


def _empty_stats() -> dict[str, Any]:
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "max_single_loss": 0.0,
        "avg_rr": 0.0,
    }


def _empty_overall_stats() -> dict[str, Any]:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl_pct": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_confidence": 0.0,
        "best_session": "",
    }


def _closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result_mask = df["result"].astype(str).isin({"WIN", "LOSS", "BE"})
    exit_mask = pd.to_datetime(df["exit_time"], utc=True, errors="coerce").notna()
    return df.loc[result_mask | exit_mask].copy()


def _win_rate_percent(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    completed = df["result"].astype(str).isin({"WIN", "LOSS", "BE"})
    completed_count = int(completed.sum())
    if completed_count == 0:
        return 0.0
    wins = int(df.loc[completed, "result"].astype(str).eq("WIN").sum())
    return round(wins / completed_count * 100.0, 2)


def _sharpe_ratio(pnl_series: pd.Series) -> float:
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if len(clean) < 2 or clean.std() == 0:
        return 0.0
    return round(float(clean.mean() / clean.std() * sqrt(252)), 4)


def _profit_factor(pnl_series: pd.Series) -> float:
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    gains = clean[clean > 0].sum()
    losses = abs(clean[clean < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return round(float(gains / losses), 4)


def _max_drawdown(pnl_series: pd.Series) -> float:
    clean = pd.to_numeric(pnl_series, errors="coerce").fillna(0.0)
    if clean.empty:
        return 0.0
    equity_curve = clean.cumsum()
    rolling_peak = equity_curve.cummax()
    drawdowns = rolling_peak - equity_curve
    return round(float(drawdowns.max()), 2)


def _resolve_starting_balance(total_pnl: float) -> float | None:
    configured = _coerce_float(getattr(config, "STARTING_BALANCE", None))
    if configured and configured > 0:
        return configured

    account_info = _get_account_snapshot()
    balance = _coerce_float(account_info.get("balance"))
    if balance is None:
        return None

    inferred = balance - total_pnl
    return inferred if inferred > 0 else None


def _format_number(value: float) -> str:
    return f"{value:,.2f}"


def _format_signed_number(value: float) -> str:
    return f"{value:+,.2f}"


def _get_account_snapshot() -> dict[str, float]:
    try:
        from data_feed import get_account_info

        info = get_account_info()
        return info or {}
    except Exception as exc:
        logger.warning(f"Unable to fetch account snapshot for dashboard: {exc}")
        return {}


def _get_open_trade_count() -> int:
    try:
        from execution import get_open_positions

        return len(get_open_positions())
    except Exception as exc:
        logger.warning(f"Unable to fetch open trades for dashboard: {exc}")
        return 0


def _update_exit(
    ticket: int,
    exit_price: float,
    exit_reason: str,
    pnl_currency_override: float | None = None,
) -> bool:
    df = _read_journal()
    if df.empty:
        logger.warning(f"Trade {ticket} not found in journal.")
        return False

    ticket_mask = pd.to_numeric(df["ticket"], errors="coerce") == int(ticket)
    matches = df.index[ticket_mask].tolist()
    if not matches:
        logger.warning(f"Trade {ticket} not found in journal.")
        return False

    open_matches = [idx for idx in matches if pd.isna(df.loc[idx, "exit_price"])]
    row_idx = open_matches[-1] if open_matches else matches[-1]

    row = df.loc[row_idx]
    entry_price = _coerce_float(row.get("entry_price"))
    lot_size = _coerce_float(row.get("lot_size")) or 0.0
    symbol = str(row.get("symbol") or config.SYMBOL)
    direction = str(row.get("direction") or "").upper()

    if entry_price is None or direction not in {"BUY", "SELL"}:
        logger.warning(f"Trade {ticket} is missing entry metadata; exit was not logged.")
        return False

    pnl_pips = _pnl_pips(entry_price, float(exit_price), direction, symbol)
    pnl_currency = (
        round(float(pnl_currency_override), 2)
        if pnl_currency_override is not None
        else _pnl_currency(pnl_pips, lot_size, symbol)
    )

    for column in ("exit_time", "result", "exit_reason"):
        df[column] = df[column].astype(object)

    df.loc[row_idx, "exit_price"] = round(float(exit_price), 5)
    df.loc[row_idx, "exit_time"] = _format_timestamp()
    df.loc[row_idx, "pnl_pips"] = pnl_pips
    df.loc[row_idx, "pnl_currency"] = pnl_currency
    df.loc[row_idx, "result"] = _result_from_pnl(pnl_currency)
    df.loc[row_idx, "exit_reason"] = str(exit_reason).upper()

    _write_journal(df)
    return True


def log_entry(trade_data: dict) -> None:
    """
    Appends a new trade entry to the journal, creating the CSV if needed.
    """
    try:
        timestamp = _parse_timestamp(trade_data.get("timestamp"))
        ticket = _coerce_int(trade_data.get("ticket"))
        if ticket is None:
            raise ValueError("trade_data must include a valid ticket")

        direction = str(trade_data.get("direction") or trade_data.get("action") or "").upper()
        if direction not in {"BUY", "SELL"}:
            raise ValueError("trade_data must include direction BUY or SELL")

        entry_price = _coerce_float(trade_data.get("entry_price"))
        sl = _coerce_float(trade_data.get("sl"))
        tp = _coerce_float(trade_data.get("tp"))
        lot_size = _coerce_float(trade_data.get("lot_size", trade_data.get("lots")))

        if entry_price is None or sl is None or tp is None or lot_size is None:
            raise ValueError("trade_data must include entry_price, sl, tp, and lot_size")

        df = _read_journal()
        if (pd.to_numeric(df["ticket"], errors="coerce") == ticket).any():
            logger.warning(f"Trade {ticket} already exists in journal. Skipping duplicate entry.")
            return

        row = {column: pd.NA for column in JOURNAL_COLUMNS}
        row.update(
            {
                "ticket": ticket,
                "timestamp": _format_timestamp(timestamp),
                "symbol": str(trade_data.get("symbol") or config.SYMBOL).upper(),
                "direction": direction,
                "entry_price": round(entry_price, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "lot_size": round(lot_size, 2),
                "confidence": _coerce_float(trade_data.get("confidence")),
                "drawdown_at_entry": _coerce_float(trade_data.get("drawdown_at_entry")),
                "session": str(trade_data.get("session") or _infer_session(timestamp)).upper(),
                "exit_price": _coerce_float(trade_data.get("exit_price")),
                "exit_time": _format_timestamp(_parse_timestamp(trade_data.get("exit_time")))
                if _has_value(trade_data.get("exit_time"))
                else pd.NA,
                "pnl_pips": _coerce_float(trade_data.get("pnl_pips")),
                "pnl_currency": _coerce_float(trade_data.get("pnl_currency")),
                "result": str(trade_data.get("result") or "").upper(),
                "exit_reason": str(trade_data.get("exit_reason") or "").upper(),
            }
        )

        if row["result"] == "" and row["pnl_currency"] is not None:
            row["result"] = _result_from_pnl(float(row["pnl_currency"]))

        updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _write_journal(updated)
        logger.info(f"Logged journal entry for ticket {ticket}.")

    except Exception as exc:
        logger.error(f"Failed to log journal entry: {exc}")


def log_exit(ticket: int, exit_price: float, exit_reason: str) -> None:
    """
    Updates the journal row for *ticket* with exit details and derived PnL.
    """
    try:
        if _update_exit(ticket=ticket, exit_price=exit_price, exit_reason=exit_reason):
            logger.info(f"Logged exit for trade {ticket}.")
    except Exception as exc:
        logger.error(f"Failed to log exit for trade {ticket}: {exc}")


def get_daily_summary() -> dict[str, Any]:
    """
    Returns summary stats for trades opened today in UTC.
    """
    try:
        df = _read_journal()
        if df.empty:
            return _empty_stats()

        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        today_utc = _timestamp_now().date()
        daily_df = df.loc[timestamps.dt.date == today_utc].copy()
        if daily_df.empty:
            return _empty_stats()

        closed = _closed_trades(daily_df)
        wins = int(closed["result"].astype(str).eq("WIN").sum()) if not closed.empty else 0
        losses = int(closed["result"].astype(str).eq("LOSS").sum()) if not closed.empty else 0
        total_pnl = round(float(pd.to_numeric(closed["pnl_currency"], errors="coerce").fillna(0.0).sum()), 2)

        pnl_values = pd.to_numeric(closed["pnl_currency"], errors="coerce").dropna()
        losses_only = pnl_values[pnl_values < 0]
        max_single_loss = round(float(losses_only.min()), 2) if not losses_only.empty else 0.0

        rr_values = closed.apply(_rr_multiple, axis=1).dropna() if not closed.empty else pd.Series(dtype=float)
        avg_rr = round(float(rr_values.mean()), 2) if not rr_values.empty else 0.0

        return {
            "total_trades": int(len(daily_df)),
            "wins": wins,
            "losses": losses,
            "win_rate": _win_rate_percent(closed),
            "total_pnl": total_pnl,
            "max_single_loss": max_single_loss,
            "avg_rr": avg_rr,
        }

    except Exception as exc:
        logger.error(f"Failed to compute daily summary: {exc}")
        return _empty_stats()


def get_overall_stats() -> dict[str, Any]:
    """
    Returns full-journal aggregate performance statistics.
    """
    try:
        df = _read_journal()
        if df.empty:
            return _empty_overall_stats()

        closed = _closed_trades(df)
        pnl_series = pd.to_numeric(closed["pnl_currency"], errors="coerce").fillna(0.0)
        total_pnl = float(pnl_series.sum())
        starting_balance = _resolve_starting_balance(total_pnl)
        total_pnl_pct = round(total_pnl / starting_balance * 100.0, 2) if starting_balance else 0.0

        confidence_series = pd.to_numeric(df["confidence"], errors="coerce").dropna()
        avg_confidence = round(float(confidence_series.mean()), 4) if not confidence_series.empty else 0.0

        best_session = ""
        if not closed.empty:
            session_pnl = (
                closed.assign(
                    pnl_currency_num=pd.to_numeric(closed["pnl_currency"], errors="coerce").fillna(0.0)
                )
                .groupby("session", dropna=True)["pnl_currency_num"]
                .sum()
            )
            if not session_pnl.empty:
                best_session = str(session_pnl.idxmax())

        return {
            "total_trades": int(len(df)),
            "win_rate": _win_rate_percent(closed),
            "total_pnl_pct": total_pnl_pct,
            "max_drawdown": _max_drawdown(pnl_series),
            "sharpe_ratio": _sharpe_ratio(pnl_series),
            "profit_factor": _profit_factor(pnl_series),
            "avg_confidence": avg_confidence,
            "best_session": best_session,
        }

    except Exception as exc:
        logger.error(f"Failed to compute overall stats: {exc}")
        return _empty_overall_stats()


def print_dashboard() -> None:
    """
    Prints a compact terminal dashboard using live account info plus journal stats.
    """
    account_info = _get_account_snapshot()
    daily_stats = get_daily_summary()
    overall_stats = get_overall_stats()
    open_count = _get_open_trade_count()

    balance = _coerce_float(account_info.get("balance")) or 0.0
    equity = _coerce_float(account_info.get("equity")) or 0.0
    drawdown_pct = _coerce_float(account_info.get("drawdown_pct")) or 0.0
    daily_pnl = _coerce_float(daily_stats.get("total_pnl")) or 0.0
    total_trades = int(overall_stats.get("total_trades", 0) or 0)
    win_rate = _coerce_float(overall_stats.get("win_rate")) or 0.0

    def row(label: str, value: str) -> str:
        return f"║ {f'{label:<14} {value}':<35} ║"

    print("╔═══════════════════════════════════════╗")
    print("║        THE5ERS ML BOT DASHBOARD      ║")
    print("╠═══════════════════════════════════════╣")
    print(row("Balance:", _format_number(balance)))
    print(row("Equity:", _format_number(equity)))
    print(row("Drawdown:", f"{drawdown_pct:.2f}%"))
    print(row("Today PnL:", _format_signed_number(daily_pnl)))
    print(row("Total Trades:", str(total_trades)))
    print(row("Win Rate:", f"{win_rate:.1f}%"))
    print(row("Open Trades:", str(open_count)))
    print("╚═══════════════════════════════════════╝")


def log_trade(
    ticket: int,
    symbol: str,
    action: str,
    lots: float,
    price: float,
    sl: float,
    tp: float,
) -> None:
    """
    Backward-compatible entry logger used by the existing trading loop.
    """
    log_entry(
        {
            "ticket": ticket,
            "symbol": symbol,
            "direction": action,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "lot_size": lots,
        }
    )


def log_closing(ticket: int, close_price: float, profit: float) -> None:
    """
    Backward-compatible exit logger used by the existing trading loop.
    """
    try:
        updated = _update_exit(
            ticket=ticket,
            exit_price=close_price,
            exit_reason="MANUAL",
            pnl_currency_override=profit,
        )
        if updated:
            logger.info(f"Logged close for trade {ticket}. Profit: {profit}")
    except Exception as exc:
        logger.error(f"Failed to log closing for trade {ticket}: {exc}")
