"""
execution.py

All MT5 order placement, position management, and emergency close logic.
Rules enforced here:
  - 2-second minimum delay between consecutive order placements
  - try/except on every MT5 call
  - All orders tagged with BOT_MAGIC_NUMBER for isolation
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Module-level timestamp of the last order send (for 2 s throttle)
_last_order_time: Optional[float] = None


def _enforce_order_delay() -> None:
    """Blocks until at least 2 seconds have elapsed since the last order."""
    global _last_order_time
    if _last_order_time is not None:
        elapsed = time.monotonic() - _last_order_time
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)


def _stamp_order_time() -> None:
    global _last_order_time
    _last_order_time = time.monotonic()


# ---------------------------------------------------------------------------
# 1. place_order
# ---------------------------------------------------------------------------

def place_order(
    symbol: str,
    signal: int,
    lot: float,
    sl_price: float,
    tp_price: float,
) -> Optional[dict]:
    """
    Places a market order in MT5.

    Args:
        symbol:   Trading symbol (e.g. "EURUSD").
        signal:   1 for Buy, -1 for Sell.
        lot:      Volume in lots.
        sl_price: Stop-loss price.
        tp_price: Take-profit price.

    Returns:
        dict with ticket, price, sl, tp, volume, time — or None on failure.
    """
    _enforce_order_delay()

    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(
                f"place_order: cannot get tick for {symbol}: {mt5.last_error()}"
            )
            return None

        price = tick.ask if signal == 1 else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lot),
            "type":         order_type,
            "price":        float(price),
            "sl":           float(sl_price),
            "tp":           float(tp_price),
            "deviation":    10,
            "magic":        config.BOT_MAGIC_NUMBER,
            "comment":      "ML_BOT_V1",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        _stamp_order_time()

        if result is None:
            logger.error(
                f"place_order: order_send returned None for {symbol}. "
                f"MT5 error: {mt5.last_error()}"
            )
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"place_order: order rejected — symbol={symbol} "
                f"signal={signal} lot={lot} "
                f"retcode={result.retcode} comment='{result.comment}' "
                f"MT5 error: {mt5.last_error()}"
            )
            return None

        logger.info(
            f"place_order: FILLED ticket={result.order} "
            f"symbol={symbol} signal={signal} lot={lot} "
            f"price={result.price} sl={sl_price} tp={tp_price}"
        )

        return {
            "ticket": result.order,
            "price":  result.price,
            "sl":     sl_price,
            "tp":     tp_price,
            "volume": float(lot),
            "time":   datetime.now(UTC),
        }

    except Exception as exc:
        logger.exception(f"place_order: unexpected error for {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 2. close_order
# ---------------------------------------------------------------------------

def close_order(ticket: int) -> bool:
    """
    Closes an open position by placing an opposing market order.

    Args:
        ticket: Position ticket number.

    Returns:
        True if closed successfully, False otherwise.
    """
    _enforce_order_delay()

    try:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"close_order: no open position found for ticket {ticket}.")
            return False

        pos = positions[0]
        symbol = pos.symbol
        lot    = pos.volume

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(
                f"close_order: cannot get tick for {symbol}: {mt5.last_error()}"
            )
            return False

        # To close a Buy we Sell; to close a Sell we Buy
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lot),
            "type":         close_type,
            "position":     ticket,
            "price":        float(price),
            "deviation":    10,
            "magic":        config.BOT_MAGIC_NUMBER,
            "comment":      "ML_BOT_V1_CLOSE",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        _stamp_order_time()

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            logger.error(
                f"close_order: failed — ticket={ticket} retcode={retcode} "
                f"MT5 error: {mt5.last_error()}"
            )
            return False

        logger.info(f"close_order: closed ticket={ticket} symbol={symbol}")
        return True

    except Exception as exc:
        logger.exception(f"close_order: unexpected error for ticket {ticket}: {exc}")
        return False


# ---------------------------------------------------------------------------
# 3. get_open_positions
# ---------------------------------------------------------------------------

def get_open_positions(symbol: Optional[str] = None) -> list[dict]:
    """
    Returns open positions belonging to this bot (BOT_MAGIC_NUMBER),
    optionally filtered by symbol.

    Returns:
        List of dicts: ticket, symbol, type, volume, open_price,
                       sl, tp, open_time, profit.
    """
    try:
        if symbol:
            raw = mt5.positions_get(symbol=symbol)
        else:
            raw = mt5.positions_get()

        if raw is None:
            logger.error(f"get_open_positions: positions_get failed: {mt5.last_error()}")
            return []

        result = []
        for pos in raw:
            if pos.magic != config.BOT_MAGIC_NUMBER:
                continue
            result.append({
                "ticket":     pos.ticket,
                "symbol":     pos.symbol,
                "type":       pos.type,        # 0=Buy, 1=Sell
                "volume":     pos.volume,
                "open_price": pos.price_open,
                "sl":         pos.sl,
                "tp":         pos.tp,
                "open_time":  datetime.fromtimestamp(pos.time, tz=UTC),
                "profit":     pos.profit,
            })

        return result

    except Exception as exc:
        logger.exception(f"get_open_positions: unexpected error: {exc}")
        return []


# ---------------------------------------------------------------------------
# 4. count_open_trades
# ---------------------------------------------------------------------------

def count_open_trades(symbol: Optional[str] = None) -> int:
    """Returns the number of open bot positions (optionally filtered by symbol)."""
    return len(get_open_positions(symbol))


# ---------------------------------------------------------------------------
# 5. check_min_duration
# ---------------------------------------------------------------------------

def check_min_duration(ticket: int) -> bool:
    """
    Returns True if the position identified by *ticket* has been open for
    at least MIN_TRADE_DURATION seconds.
    """
    try:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"check_min_duration: ticket {ticket} not found.")
            return False

        pos = positions[0]
        open_time = datetime.fromtimestamp(pos.time, tz=UTC)
        elapsed = (datetime.now(UTC) - open_time).total_seconds()

        if elapsed >= config.MIN_TRADE_DURATION:
            return True

        logger.info(
            f"check_min_duration: ticket={ticket} open for {elapsed:.0f}s "
            f"(min={config.MIN_TRADE_DURATION}s) — not eligible to close yet."
        )
        return False

    except Exception as exc:
        logger.exception(
            f"check_min_duration: unexpected error for ticket {ticket}: {exc}"
        )
        return False


# ---------------------------------------------------------------------------
# 6. close_all_positions
# ---------------------------------------------------------------------------

def close_all_positions() -> int:
    """
    Emergency-closes every open position belonging to this bot.
    Inserts a 2-second delay between each close to avoid broker throttling.

    Returns:
        Number of positions successfully closed.
    """
    positions = get_open_positions()
    if not positions:
        logger.info("close_all_positions: no open positions to close.")
        return 0

    logger.warning(
        f"close_all_positions: emergency-closing {len(positions)} position(s)."
    )

    closed = 0
    for pos in positions:
        if close_order(pos["ticket"]):
            closed += 1
            logger.info(f"close_all_positions: closed ticket={pos['ticket']}")
        else:
            logger.error(
                f"close_all_positions: FAILED to close ticket={pos['ticket']}"
            )
        # 2-second delay already enforced inside close_order via _enforce_order_delay,
        # but we add an explicit sleep here as belt-and-braces for the emergency path.
        time.sleep(2)

    logger.warning(
        f"close_all_positions: {closed}/{len(positions)} closed successfully."
    )
    return closed
