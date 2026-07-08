"""
main.py

Live trading loop for the The5ers Bootcamp rules bot.

Signal flow (rules-first — the ML model can only veto, never create):
  strategy.py candidate → entry filters (session/news/spread/volatility)
  → optional MetaVeto → RiskManager (kill switch, weekly/daily stops,
  SL clamp, sizing) → execution (mandatory broker-visible SL).

Runs on an M15 cadence; every candle it also manages open positions
(breakeven move at +1R, flatten before major news). Supports dry-run mode.
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import config
import data_feed
import execution
import news_filter
import strategy
import trade_journal
from symbol_specs import get_symbol_spec

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    filename=config.LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Also log to stdout so the operator can watch
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)

UTC = timezone.utc

SYMBOL = config.SYMBOL

# Bars needed for indicator warm-up (EMA200 on H1; EMA/ATR/median on M15)
M15_BARS = 2200   # ≈ 23 days: enough for the 20-day ATR-median filter to form
H1_BARS = 400


# ── BotState ──────────────────────────────────────────────────

@dataclass
class BotState:
    starting_balance: float
    risk: Any                        # RiskManager (lazy import avoids cycles)
    meta_veto: Optional[Any] = None  # model.MetaVeto when USE_META_VETO
    params: strategy.StrategyParams = field(default_factory=strategy.StrategyParams)
    is_running: bool = True


# ── Helpers ───────────────────────────────────────────────────

def get_current_m15_time() -> datetime:
    """Round current UTC time **down** to the nearest M15 boundary."""
    now = datetime.now(UTC)
    minute = (now.minute // 15) * 15
    return now.replace(minute=minute, second=0, microsecond=0)


# ── Initialize ────────────────────────────────────────────────

def initialize_bot(dry_run: bool = False) -> BotState:
    """
    Connects to the configured broker, creates the RiskManager, optionally
    loads the MetaVeto model, prints a startup banner, and returns BotState.
    """
    from risk_manager import RiskManager

    # 1. Connect to broker and obtain starting balance
    starting_balance = data_feed.connect_broker()
    logger.info(f"Connected to {config.BROKER} — starting balance: {starting_balance}")

    # 2. Risk manager (loads any persisted disabled flag) + journal counters
    risk = RiskManager(starting_balance)
    risk.sync_counters_from_journal()

    # 3. Optional MetaVeto (may only BLOCK candidates; off by default)
    meta_veto = None
    if config.USE_META_VETO:
        try:
            from model import MetaVeto

            meta_veto = MetaVeto.load()
            logger.info("MetaVeto model loaded (blocking-only filter enabled).")
        except Exception as exc:
            logger.warning(f"MetaVeto unavailable ({exc}) — continuing rules-only.")

    mode_label = "DRY-RUN" if dry_run else "LIVE"
    banner = (
        "\n"
        "╔══════════════════════════════════════╗\n"
        "║   THE5ERS BOOTCAMP RULES BOT v2.0   ║\n"
        f"║    Symbol: {SYMBOL:<7s} |  TF: M15       ║\n"
        f"║    Broker: {config.BROKER:<25s}║\n"
        f"║    Mode: {mode_label:<27s}║\n"
        f"║    Balance: {starting_balance:<24.2f}║\n"
        f"║    MetaVeto: {'ON' if meta_veto else 'OFF':<23s}║\n"
        "╚══════════════════════════════════════╝"
    )
    print(banner)
    logger.info(banner)

    return BotState(
        starting_balance=starting_balance,
        risk=risk,
        meta_veto=meta_veto,
    )


# ── Position management (every candle, before entries) ───────

def manage_positions(state: BotState, dry_run: bool = False) -> None:
    """Breakeven move at +1R and the flatten-before-major-news rule."""
    try:
        open_positions = execution.get_open_positions(SYMBOL)
    except Exception as exc:
        logger.exception(f"manage_positions: failed to read positions: {exc}")
        return
    if not open_positions:
        return

    try:
        if news_filter.should_flatten_for_news(SYMBOL):
            logger.warning("manage_positions: MAJOR news imminent — flattening.")
            if not dry_run:
                execution.close_all_positions()
            return
    except Exception:
        logger.warning("manage_positions: news check failed — flattening (fail closed).")
        if not dry_run:
            execution.close_all_positions()
        return

    if not dry_run:
        execution.manage_breakeven(SYMBOL)


# ── Candle processor ──────────────────────────────────────────

def process_candle(state: BotState, dry_run: bool = False) -> None:
    """
    Runs once per new M15 candle.  Steps:
      1. Equity check + open-position management (BE move, news flatten)
      2. Risk counters from journal + can_trade gate
      3. Session filter (Europe/London, DST-aware)
      4. News entry blackout (fail closed)
      5. Rules candidate from strategy.py
      6. Spread + volatility filters
      7. Optional MetaVeto (block-only)
      8. SL/TP via clamp (skip when violated) + lot sizing
      9. Order execution (or dry-run log) + journal + counters
    """

    # Step 1: current equity (includes floating P&L) and position upkeep
    try:
        acct = data_feed.get_account_info()
        if not acct:
            logger.error("process_candle: broker account info returned empty.")
            return
        equity = float(acct["equity"])
    except Exception as exc:
        logger.exception(f"process_candle: failed to read equity: {exc}")
        return

    manage_positions(state, dry_run=dry_run)

    # Step 2: risk gate (kill switch, weekly/daily stops, pacing, rollover)
    try:
        state.risk.sync_counters_from_journal()
    except Exception as exc:
        logger.warning(f"process_candle: journal sync failed: {exc}")

    open_trades = execution.count_open_trades(SYMBOL)
    can_trade, reason = state.risk.can_trade(equity, open_trades)
    if not can_trade:
        logger.warning(f"process_candle: blocked — {reason}")
        return

    # Step 3: session window (entries 08:00–17:00 London, Friday cutoff, …)
    now = datetime.now(UTC)
    if not strategy.entry_session_ok(now):
        logger.info("process_candle: outside entry session — skipping.")
        return

    # Step 4: news entry blackout (±30 min high impact, ±60 min majors)
    try:
        if news_filter.is_news_window(SYMBOL):
            logger.info("process_candle: inside news window — skipping.")
            return
    except Exception:
        logger.warning("process_candle: news filter failed — skipping candle (fail-closed).")
        return

    # Step 5: rules candidate (the strategy IS the rules)
    try:
        df_m15 = data_feed.get_ohlcv(SYMBOL, "M15", M15_BARS)
        df_h1 = data_feed.get_ohlcv(SYMBOL, "H1", H1_BARS)
        params = strategy.StrategyParams(adx_min=state.risk.adx_min_for_next_trade())
        candidate = strategy.generate_candidate(df_m15, df_h1, params)
    except Exception as exc:
        logger.exception(f"process_candle: candidate generation failed: {exc}")
        return
    if candidate is None:
        logger.info("process_candle: no rules setup — no trade.")
        return

    # Step 6: spread + volatility filters
    spec = get_symbol_spec(SYMBOL)
    try:
        tick = data_feed.get_latest_tick(SYMBOL)
        spread_pips = (float(tick["ask"]) - float(tick["bid"])) / spec.pip_size
    except Exception as exc:
        logger.exception(f"process_candle: tick fetch failed: {exc}")
        return
    if not strategy.spread_ok(spread_pips):
        logger.info(f"process_candle: spread {spread_pips:.1f} pips too wide — skipping.")
        return

    atr_pips = candidate.atr / spec.pip_size
    atr_median_pips = (
        candidate.atr_median / spec.pip_size if candidate.atr_median else None
    )
    if not strategy.volatility_ok(atr_pips, atr_median_pips):
        logger.info(f"process_candle: ATR {atr_pips:.1f} pips outside band — skipping.")
        return

    # Step 7: optional MetaVeto — it may only BLOCK, never create
    if state.meta_veto is not None:
        try:
            import features

            X_live = features.get_live_features(SYMBOL)
            if not X_live.empty and not state.meta_veto.allow(
                candidate.direction, X_live.to_frame().T
            ):
                logger.info("process_candle: candidate blocked by MetaVeto.")
                return
        except Exception as exc:
            logger.warning(f"process_candle: MetaVeto failed open ({exc}).")

    # Step 8: SL/TP (clamp skip) + lot sizing (heartbeat-aware risk %)
    entry = float(tick["ask"]) if candidate.direction == 1 else float(tick["bid"])
    sl_tp = state.risk.calculate_sl_tp(
        candidate.direction, entry, candidate.atr, swing_price=candidate.swing_price
    )
    if sl_tp is None:
        logger.info("process_candle: SL outside [8, 25] pip clamp — skipping trade.")
        return
    sl, tp = sl_tp
    lot = state.risk.calculate_lot_size(equity, sl, entry, SYMBOL)
    if lot <= 0:
        logger.info("process_candle: lot size 0 — skipping trade.")
        return

    # Step 9: Execute or dry-run
    action = "BUY" if candidate.direction == 1 else "SELL"
    if dry_run:
        logger.info(
            f"[DRY RUN] {action} candidate ({candidate.reason}) "
            f"entry={entry} sl={sl} tp={tp} lot={lot}"
        )
        return

    result = execution.place_order(SYMBOL, candidate.direction, lot, sl, tp)
    if result:
        trade_journal.log_trade(
            ticket=result["ticket"],
            symbol=SYMBOL,
            action=action,
            lots=lot,
            price=result["price"],
            sl=sl,
            tp=tp,
        )
        state.risk.record_trade_opened()
        logger.info(
            f"TRADE OPENED — ticket={result['ticket']} "
            f"{action} {lot} lots @ {result['price']}  SL={sl}  TP={tp}"
        )
    else:
        logger.error("process_candle: place_order returned None — order failed.")


# ── Main entry ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="The5ers Bootcamp Rules Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (log signals but do not place real orders).",
    )
    args = parser.parse_args()

    state = initialize_bot(dry_run=args.dry_run)
    last_candle: datetime | None = None

    logger.info("Entering main loop — polling every 5 s ...")

    while state.is_running:
        try:
            candle_time = get_current_m15_time()
            if candle_time != last_candle:
                logger.info(f"New M15 candle: {candle_time.isoformat()}")
                last_candle = candle_time
                process_candle(state, dry_run=args.dry_run)

            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Shutting down (KeyboardInterrupt) ...")
            data_feed.shutdown_broker()
            break

        except Exception as exc:
            logger.exception(f"Unhandled error in main loop: {exc}")
            time.sleep(5)


if __name__ == "__main__":
    main()
