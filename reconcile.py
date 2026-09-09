"""
reconcile.py

Finding 2 fix: reconcile broker-side closes (SL/TP hit while the bot was
between candles or offline) back into the trade journal.

The live loop only ever called ``log_trade`` on entry; when the broker closed
a position at SL/TP nothing recorded the exit, so ``sync_counters_from_journal``
saw no results and the consecutive-loss daily/weekly stops never fired. This
module detects journalled-open tickets that are no longer open at the broker,
classifies each close, writes the exit to the journal, and feeds the result
into the RiskManager's streak counters.

``classify_closure`` is pure (no broker, no I/O) so it is fully unit-testable.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


def classify_closure(
    journal_row: dict,
    last_profit: float | None,
) -> tuple[float, str, str]:
    """
    Decide the (exit_price, exit_reason, result) for a ticket that is open in
    the journal but no longer open at the broker.

    We do not assume access to broker deal history (cTrader adapter lacks it),
    so we reconstruct the exit from what we DO know: the entry/SL/TP recorded
    at order time, plus the last-seen floating P&L captured on the previous
    loop. SL/TP orders execute AT their levels, so using SL or TP as the exit
    price is more accurate than a stale mid, and the sign of the last-seen P&L
    tells us which barrier fired.

    Fallbacks are deliberately conservative: when the outcome is unknown we
    assume an SL loss, so the consecutive-loss stop errs toward halting.
    """
    entry = journal_row.get("entry_price")
    direction = str(journal_row.get("direction") or "").upper()
    sl = journal_row.get("sl")
    tp = journal_row.get("tp")

    # Is this a winner? Prefer the last-seen broker P&L; else infer nothing.
    is_win: bool | None
    if last_profit is None:
        is_win = None
    elif last_profit > 0:
        is_win = True
    else:
        is_win = False

    # Choose the exit price and reason.
    if is_win is True and tp is not None:
        exit_price, reason = float(tp), "TP"
    elif is_win is False and sl is not None:
        exit_price, reason = float(sl), "SL"
    elif sl is not None:
        # Unknown outcome → conservative assumed-SL loss.
        exit_price, reason = float(sl), "RECONCILED_ASSUMED_SL"
        is_win = False
    elif entry is not None:
        exit_price, reason = float(entry), "RECONCILED_FLAT"
        is_win = None
    else:
        # No metadata at all — cannot reconstruct; signal caller to skip.
        raise ValueError("classify_closure: journal row lacks entry/SL/TP")

    # Breakeven detection: an SL that was trailed to the entry price is a
    # scratch, not a loss — leave the streak untouched.
    result: str
    if (
        entry is not None
        and reason in {"SL", "RECONCILED_ASSUMED_SL"}
        and abs(float(exit_price) - float(entry)) <= 1e-9
    ):
        result, reason = "BE", "BREAKEVEN"
    elif is_win is True:
        result = "WIN"
    elif is_win is False:
        result = "LOSS"
    else:
        result = "LOSS"  # conservative default

    return exit_price, reason, result


def reconcile_closures(
    journal_open: Iterable[dict],
    broker_open_tickets: set[int],
    last_profit_by_ticket: dict[int, float],
    on_exit: Callable[[int, float, str], None],
    on_result: Callable[[str], None],
) -> list[dict]:
    """
    For every journalled-open ticket that is NOT in ``broker_open_tickets``,
    classify the close, persist the exit via ``on_exit(ticket, price, reason)``
    and update streak counters via ``on_result(result)``.

    Returns a list of reconciliation records (for logging/tests). All callbacks
    are injected so this is testable without a broker or the journal CSV.
    """
    reconciled: list[dict] = []
    for row in journal_open:
        ticket = row.get("ticket")
        if ticket is None or int(ticket) in broker_open_tickets:
            continue
        ticket = int(ticket)
        try:
            exit_price, reason, result = classify_closure(
                row, last_profit_by_ticket.get(ticket)
            )
        except ValueError as exc:
            logger.warning("reconcile: skipping ticket %s — %s", ticket, exc)
            continue

        try:
            on_exit(ticket, exit_price, reason)
            on_result(result)
        except Exception as exc:  # never let reconciliation crash the loop
            logger.exception("reconcile: callback failed for ticket %s: %s", ticket, exc)
            continue

        logger.info(
            "Reconciled broker-side close: ticket=%s exit=%.5f reason=%s result=%s",
            ticket, exit_price, reason, result,
        )
        reconciled.append(
            {"ticket": ticket, "exit_price": exit_price, "reason": reason, "result": result}
        )
    return reconciled
