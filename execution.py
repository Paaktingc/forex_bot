"""
execution.py

Order placement, position management, and emergency close logic.
Rules enforced here:
  - EVERY order must carry a broker-visible stop loss (The5ers requirement);
    orders without a valid SL are rejected here, not just in the strategy
  - market orders only — pending orders (incl. straddles around news) are
    not supported by design
  - 2-second minimum delay between consecutive order placements
  - SL modifications (breakeven moves) are rate-limited per ticket and
    applied at most once (The5ers bans EAs that spam order modifications)
  - try/except on broker calls
  - MT5 orders tagged with BOT_MAGIC_NUMBER for isolation
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available — Mac/dev mode")

import config
from brokers import get_broker

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Module-level timestamp of the last order send (for 2 s throttle)
_last_order_time: Optional[float] = None

# Breakeven bookkeeping: tickets already moved + per-ticket last SL change
_be_applied_tickets: set[int] = set()
_last_modify_time: dict[int, float] = {}


def _valid_stop_loss(signal: int, sl_price, tp_price) -> bool:
    """
    A broker-visible SL is mandatory on every order. Validates that sl_price
    is a positive finite number on the LOSS side of the trade (below TP for
    buys, above TP for sells).
    """
    try:
        sl = float(sl_price)
        tp = float(tp_price)
    except (TypeError, ValueError):
        return False
    if not (sl > 0 and sl == sl and tp == tp):  # NaN-safe
        return False
    if signal == 1:
        return sl < tp
    if signal == -1:
        return sl > tp
    return False


def _require_mt5() -> None:
    if not MT5_AVAILABLE:
        raise RuntimeError(
            "MT5 not available. Deploy to Windows VPS for live trading."
        )


def _use_broker_adapter() -> bool:
    return config.BROKER == "ctrader"


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
    Places a market order through the configured broker.

    Args:
        symbol:   Trading symbol (e.g. "EURUSD").
        signal:   1 for Buy, -1 for Sell.
        lot:      Volume in lots.
        sl_price: Stop-loss price.
        tp_price: Take-profit price.

    Returns:
        dict with ticket, price, sl, tp, volume, time — or None on failure.
    """
    if not _valid_stop_loss(signal, sl_price, tp_price):
        logger.error(
            f"place_order REJECTED: order without a valid broker-visible SL "
            f"(symbol={symbol} signal={signal} sl={sl_price} tp={tp_price}). "
            f"The5ers requires a stop loss on every position."
        )
        return None

    if _use_broker_adapter():
        return get_broker().place_order(symbol, signal, lot, sl_price, tp_price)

    _require_mt5()
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
            "comment":      config.BOT_LABEL,
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
    if _use_broker_adapter():
        return get_broker().close_order(ticket)

    _require_mt5()
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
            "comment":      f"{config.BOT_LABEL}_CLOSE",
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
    if _use_broker_adapter():
        return get_broker().get_open_positions(symbol)

    _require_mt5()
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
    if _use_broker_adapter():
        return get_broker().count_open_trades(symbol)
    return len(get_open_positions(symbol))


# ---------------------------------------------------------------------------
# 5. check_min_duration
# ---------------------------------------------------------------------------

def check_min_duration(ticket: int) -> bool:
    """
    Returns True if the position identified by *ticket* has been open for
    at least MIN_TRADE_DURATION seconds.
    """
    if _use_broker_adapter():
        return get_broker().check_min_duration(ticket)

    _require_mt5()
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
# 6. modify_position_sl / manage_breakeven
# ---------------------------------------------------------------------------

def modify_position_sl(ticket: int, new_sl: float) -> bool:
    """
    Moves the stop loss of an open position. Rate-limited per ticket
    (config.BE_MODIFY_MIN_INTERVAL_S) so the bot never spams modifications.
    The SL is never removed — new_sl must be a positive price.
    """
    if not new_sl or new_sl <= 0:
        logger.error(f"modify_position_sl: refusing to set invalid SL {new_sl}.")
        return False

    now = time.monotonic()
    last = _last_modify_time.get(ticket)
    if last is not None and now - last < config.BE_MODIFY_MIN_INTERVAL_S:
        logger.info(
            f"modify_position_sl: rate limit — ticket {ticket} modified "
            f"{now - last:.0f}s ago (min {config.BE_MODIFY_MIN_INTERVAL_S}s)."
        )
        return False

    if _use_broker_adapter():
        broker = get_broker()
        modify = getattr(broker, "modify_position_sl", None)
        if modify is None:
            logger.warning(
                "modify_position_sl: broker adapter has no SL modification "
                "support — skipping breakeven move."
            )
            return False
        ok = bool(modify(ticket, new_sl))
        if ok:
            _last_modify_time[ticket] = now
        return ok

    _require_mt5()
    try:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"modify_position_sl: ticket {ticket} not found.")
            return False
        pos = positions[0]

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       float(new_sl),
            "tp":       float(pos.tp),
            "magic":    config.BOT_MAGIC_NUMBER,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            logger.error(
                f"modify_position_sl: failed — ticket={ticket} retcode={retcode} "
                f"MT5 error: {mt5.last_error()}"
            )
            return False

        _last_modify_time[ticket] = now
        logger.info(f"modify_position_sl: ticket={ticket} SL → {new_sl}")
        return True

    except Exception as exc:
        logger.exception(f"modify_position_sl: unexpected error for {ticket}: {exc}")
        return False


def manage_breakeven(symbol: Optional[str] = None) -> int:
    """
    Once a position is +BE_AT_R (1.0R) in profit, moves its SL to the entry
    price. Applied at most ONCE per ticket and rate-limited via
    modify_position_sl. Returns the number of positions moved.
    """
    from data_feed import get_latest_tick

    moved = 0
    for pos in get_open_positions(symbol):
        ticket = int(pos["ticket"])
        if ticket in _be_applied_tickets:
            continue

        entry = float(pos["open_price"])
        sl = float(pos["sl"] or 0.0)
        if sl <= 0:
            logger.error(
                f"manage_breakeven: position {ticket} has NO stop loss — "
                f"this should be impossible; skipping."
            )
            continue

        is_buy = int(pos["type"]) == 0
        risk = (entry - sl) if is_buy else (sl - entry)
        if risk <= 0:
            continue  # SL already at/beyond breakeven

        try:
            tick = get_latest_tick(pos["symbol"])
            current = float(tick["bid"]) if is_buy else float(tick["ask"])
        except Exception as exc:
            logger.warning(f"manage_breakeven: no tick for {pos['symbol']}: {exc}")
            continue

        trigger = entry + config.BE_AT_R * risk if is_buy else entry - config.BE_AT_R * risk
        reached = current >= trigger if is_buy else current <= trigger
        if not reached:
            continue

        if modify_position_sl(ticket, entry):
            _be_applied_tickets.add(ticket)
            moved += 1
            logger.info(
                f"manage_breakeven: ticket={ticket} reached +{config.BE_AT_R}R — "
                f"SL moved to breakeven {entry}."
            )

    return moved


# ---------------------------------------------------------------------------
# 7. close_all_positions
# ---------------------------------------------------------------------------

def close_all_positions() -> int:
    """
    Emergency-closes every open position belonging to this bot.
    Inserts a 2-second delay between each close to avoid broker throttling.

    Returns:
        Number of positions successfully closed.
    """
    if _use_broker_adapter():
        return get_broker().close_all_positions()

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
