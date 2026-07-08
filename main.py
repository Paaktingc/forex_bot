"""
main.py

Live trading loop for The5ers ML Forex Bot.
Coordinates data fetching, feature engineering, ML prediction,
risk management, and order execution on an M15 cadence.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sklearn.preprocessing import LabelEncoder

import config
import data_feed
import execution
import features
import model as model_module
import news_filter
import trade_journal

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


# ── BotState ──────────────────────────────────────────────────

@dataclass
class BotState:
    starting_balance: float
    model: Any
    label_encoder: LabelEncoder
    risk: object  # RiskManager (imported lazily to avoid circular deps)
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
    Connects to the configured broker, loads model + label encoder, creates RiskManager,
    prints a startup banner, and returns BotState.
    """
    from risk_manager import RiskManager

    # 1. Connect to broker and obtain starting balance
    starting_balance = data_feed.connect_broker()
    logger.info(f"Connected to {config.BROKER} — starting balance: {starting_balance}")

    # 2. Load model + label encoder
    xgb_model, label_enc = model_module.load_model()
    logger.info("Model and LabelEncoder loaded successfully.")

    # 3. Initialise RiskManager
    risk = RiskManager(starting_balance)

    # 4. Determine mode label
    mode_label = "DRY-RUN" if dry_run else "LIVE"

    # 5. Print startup banner
    banner = (
        "\n"
        "╔══════════════════════════════════════╗\n"
        "║    THE5ERS ML FOREX BOT v1.0        ║\n"
        f"║    Symbol: {SYMBOL:<7s} |  TF: M15       ║\n"
        f"║    Broker: {config.BROKER:<25s}║\n"
        f"║    Mode: {mode_label:<27s}║\n"
        f"║    Balance: {starting_balance:<24.2f}║\n"
        "╚══════════════════════════════════════╝"
    )
    print(banner)
    logger.info(banner)

    return BotState(
        starting_balance=starting_balance,
        model=xgb_model,
        label_encoder=label_enc,
        risk=risk,
    )


# ── Candle processor ──────────────────────────────────────────

def process_candle(state: BotState, dry_run: bool = False) -> None:
    """
    Runs once per new M15 candle.  Steps:
      1. Equity check
      2. can_trade gate (drawdown, daily loss, rollover, max trades)
      3. News filter
      4. Feature generation
      5. ML prediction
      6. SL/TP + lot sizing
      7. Order execution (or dry-run log)
    """

    # Step 1: current equity
    try:
        acct = data_feed.get_account_info()
        if not acct:
            logger.error("process_candle: broker account info returned empty.")
            return
        equity = float(acct["equity"])
    except Exception as exc:
        logger.exception(f"process_candle: failed to read equity: {exc}")
        return

    # Step 2: can_trade gate
    open_trades = execution.count_open_trades(SYMBOL)
    can_trade, reason = state.risk.can_trade(equity, open_trades)
    if not can_trade:
        logger.warning(f"process_candle: blocked — {reason}")
        return

    # Step 3: news window
    try:
        if news_filter.is_news_window(SYMBOL):
            logger.info("process_candle: inside news window — skipping.")
            return
    except Exception:
        logger.warning("process_candle: news filter failed — skipping candle (fail-closed).")
        return

    # Step 4: live features
    try:
        X_live = features.get_live_features(SYMBOL)
        if X_live is None or X_live.empty:
            logger.warning("process_candle: no live features available.")
            return
    except Exception as exc:
        logger.exception(f"process_candle: feature generation failed: {exc}")
        return

    # Step 5: ML prediction
    # predict_signal expects a DataFrame row, so reshape the Series
    X_df = X_live.to_frame().T
    signal, confidence = model_module.predict_signal(
        state.model, state.label_encoder, X_df
    )
    if signal == 0:
        logger.info(
            f"process_candle: signal=HOLD confidence={confidence:.2f} — no trade."
        )
        return

    # Step 6: SL/TP + lot sizing
    try:
        # Fetch a small M15 window to compute ATR for SL/TP
        df_raw = data_feed.get_ohlcv(SYMBOL, "M15", 20)
        if df_raw is None or df_raw.empty:
            logger.error("process_candle: failed to fetch OHLCV for ATR.")
            return

        df_ind = features.compute_indicators(df_raw)
        atr = float(df_ind["atr_14"].iloc[-1])

        tick = data_feed.get_latest_tick(SYMBOL)
        entry = tick["ask"] if signal == 1 else tick["bid"]

        sl_tp = state.risk.calculate_sl_tp(signal, entry, atr)
        if sl_tp is None:
            logger.info("process_candle: SL outside clamp — skipping trade.")
            return
        sl, tp = sl_tp
        lot = state.risk.calculate_lot_size(equity, sl, entry, SYMBOL)
    except Exception as exc:
        logger.exception(f"process_candle: SL/TP/lot calculation failed: {exc}")
        return

    # Step 7: Execute or dry-run
    if dry_run:
        logger.info(
            f"[DRY RUN] signal={signal} conf={confidence:.2f} "
            f"entry={entry} sl={sl} tp={tp} lot={lot}"
        )
        return

    result = execution.place_order(SYMBOL, signal, lot, sl, tp)
    if result:
        action = "BUY" if signal == 1 else "SELL"
        trade_journal.log_trade(
            ticket=result["ticket"],
            symbol=SYMBOL,
            action=action,
            lots=lot,
            price=result["price"],
            sl=sl,
            tp=tp,
        )
        logger.info(
            f"TRADE OPENED — ticket={result['ticket']} "
            f"{action} {lot} lots @ {result['price']}  SL={sl}  TP={tp}"
        )
    else:
        logger.error("process_candle: place_order returned None — order failed.")


# ── Main entry ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="The5ers ML Forex Bot")
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
    
