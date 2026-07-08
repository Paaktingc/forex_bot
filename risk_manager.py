"""
risk_manager.py

Enforces The5ers Bootcamp risk rules via RiskManager class:
  - Kill switch                    (−3% equity incl. floating from initial step
                                    balance → flatten, disable, persist to disk)
  - Official drawdown backstop     (−5% static; must never be the working limit)
  - Weekly stop                    (−1.5% from week-start balance or 5
                                    consecutive losses → halt until Monday)
  - Daily stop                     (−0.75% from day start, 2 trades, or 2
                                    consecutive losses → halt until midnight)
  - Position sizing                (0.3% risk per trade, lots floored to 0.01)
  - SL/TP calculation              (SL 1.5×ATR beyond pullback swing clamped to
                                    8–25 pips or skip; TP = 2R; BE at +1R)
  - Inactivity heartbeat           (no trade in 21 days → permit one setup at
                                    0.1% risk, ADX gate relaxed one notch)
  - can_trade() gate               (combines all checks)

Daily/weekly resets run on SERVER time (config.SERVER_TZ); the disabled flag
survives restarts and requires manual re-arm (delete the flag file).
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from symbol_specs import get_symbol_spec

logger = logging.getLogger(__name__)

UTC = timezone.utc


def _server_tz() -> ZoneInfo:
    return ZoneInfo(config.SERVER_TZ)


def server_now() -> datetime:
    return datetime.now(_server_tz())


def is_rollover_window() -> bool:
    """
    Returns True if the current UTC time falls inside the rollover window
    (ROLLOVER_START_UTC <= hour < ROLLOVER_END_UTC).
    """
    now_hour = datetime.now(UTC).hour
    return config.ROLLOVER_START_UTC <= now_hour < config.ROLLOVER_END_UTC


def is_no_trade_server_window(now: datetime | None = None) -> bool:
    """
    True inside the server-time no-trade window (NO_TRADE_SERVER_WINDOW,
    default 21:45–00:15) which spans the day boundary.
    """
    if now is None:
        now = server_now()
    else:
        now = now.astimezone(_server_tz())
    (sh, sm), (eh, em) = config.NO_TRADE_SERVER_WINDOW
    minutes = now.hour * 60 + now.minute
    start = sh * 60 + sm
    end = eh * 60 + em
    if start <= end:
        return start <= minutes < end
    return minutes >= start or minutes < end


def compute_sl_tp(
    signal: int,
    entry_price: float,
    atr: float,
    swing_price: float | None = None,
    symbol: str | None = None,
) -> tuple[float, float] | None:
    """
    Stateless SL/TP math (shared by RiskManager and the backtester).

    SL = SL_ATR_MULT (1.5) × ATR beyond the pullback swing (falls back to the
    entry price when no swing is supplied). The resulting SL distance must be
    inside [SL_MIN_PIPS, SL_MAX_PIPS]; otherwise the trade is SKIPPED
    (returns None) — the stop is never widened or narrowed to fit.

    TP = TP_R (2.0) × the ACTUAL SL distance (an R multiple, not ATR).
    """
    if atr is None or atr <= 0:
        return None

    spec = get_symbol_spec(symbol or config.SYMBOL)
    pip_size = spec.pip_size if spec else 0.0001

    anchor = swing_price if swing_price is not None else entry_price
    if signal == 1:   # Buy: SL below the pullback swing low
        sl = anchor - config.SL_ATR_MULT * atr
        sl_distance = entry_price - sl
    else:             # Sell: SL above the pullback swing high
        sl = anchor + config.SL_ATR_MULT * atr
        sl_distance = sl - entry_price

    if sl_distance <= 0:
        logger.warning("compute_sl_tp: non-positive SL distance — skipping trade.")
        return None

    sl_pips = sl_distance / pip_size
    if sl_pips < config.SL_MIN_PIPS or sl_pips > config.SL_MAX_PIPS:
        logger.info(
            f"compute_sl_tp: SL {sl_pips:.1f} pips outside "
            f"[{config.SL_MIN_PIPS}, {config.SL_MAX_PIPS}] clamp — skipping trade."
        )
        return None

    if signal == 1:
        tp = entry_price + config.TP_R * sl_distance
    else:
        tp = entry_price - config.TP_R * sl_distance

    return round(sl, 5), round(tp, 5)


class RiskManager:
    """Stateful risk manager for The5ers Bootcamp forex bot."""

    def __init__(
        self,
        starting_balance: float,
        disabled_flag_path: str | Path | None = None,
    ) -> None:
        self.starting_balance: float = starting_balance
        self.disabled_flag_path = Path(
            disabled_flag_path or config.RISK_DISABLED_FLAG_PATH
        )

        now = server_now()
        self.created_at: datetime = now
        self.daily_start_balance: float = starting_balance
        self.daily_start_time: datetime = now
        self.week_start_balance: float = starting_balance
        self.week_start_time: datetime = now
        self.halted_today: bool = False
        self.halted_this_week: bool = False

        # Pacing counters (fed from trade_journal via sync_counters_from_journal
        # in live mode, or record_trade_* in backtests)
        self.trades_today: int = 0
        self.consec_losses_day: int = 0
        self.consec_losses_week: int = 0
        self.last_trade_time: datetime | None = None

        # Martingale prevention: track last lot size
        self._last_lot_size: float = 0.0
        self._last_equity_at_lot: float = starting_balance

        if self.is_disabled():
            logger.critical(
                "RiskManager started with persisted DISABLED flag at %s — "
                "trading stays halted until the flag file is removed manually.",
                self.disabled_flag_path,
            )

    # ------------------------------------------------------------------
    # Kill switch persistence
    # ------------------------------------------------------------------

    def is_disabled(self) -> bool:
        """True when the persisted disabled flag exists (survives restarts)."""
        return self.disabled_flag_path.exists()

    def disable_trading(self, reason: str, current_equity: float | None = None) -> None:
        """Flatten everything and persist a disabled flag requiring manual re-arm."""
        logger.critical("DISABLING TRADING: %s", reason)
        try:
            from execution import close_all_positions

            close_all_positions()
        except Exception as exc:  # still persist the flag even if flatten fails
            logger.exception("disable_trading: flatten failed: %s", exc)

        payload = {
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
            "starting_balance": self.starting_balance,
            "equity": current_equity,
            "re_arm": "Delete this file to re-arm trading (manual review required).",
        }
        self.disabled_flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.disabled_flag_path.write_text(json.dumps(payload, indent=2))

    def check_kill_switch(self, current_equity: float) -> bool:
        """
        Operative hard stop. Returns False (and flattens + disables) once
        equity INCLUDING floating P&L is down KILL_SWITCH_PCT (−3%) from the
        initial step balance. The official −5% limit must never be reached.
        """
        if self.is_disabled():
            return False
        dd_pct = self.current_drawdown_pct(current_equity)
        if dd_pct >= config.KILL_SWITCH_PCT:
            self.disable_trading(
                f"KILL SWITCH: equity drawdown {dd_pct:.2%} >= "
                f"{config.KILL_SWITCH_PCT:.2%} of initial step balance",
                current_equity=current_equity,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self, current_equity: float) -> None:
        """
        Resets daily counters when a new SERVER day starts (server midnight).
        """
        now = server_now()
        if now.date() > self.daily_start_time.astimezone(_server_tz()).date():
            logger.info(
                f"New server day detected. Resetting daily balance from "
                f"{self.daily_start_balance} to current equity {current_equity}."
            )
            self.daily_start_time = now
            self.daily_start_balance = current_equity
            self.halted_today = False
            self.trades_today = 0
            self.consec_losses_day = 0
        self._maybe_reset_weekly(current_equity, now)

    def _maybe_reset_weekly(self, current_equity: float, now: datetime | None = None) -> None:
        """Resets weekly counters on Monday (server time, ISO week change)."""
        if now is None:
            now = server_now()
        start = self.week_start_time.astimezone(_server_tz())
        if now.isocalendar()[:2] != start.isocalendar()[:2] and now > start:
            logger.info(
                f"New server week detected. Resetting weekly balance from "
                f"{self.week_start_balance} to current equity {current_equity}."
            )
            self.week_start_time = now
            self.week_start_balance = current_equity
            self.halted_this_week = False
            self.consec_losses_week = 0

    # ------------------------------------------------------------------
    # Counters (fed from trade_journal)
    # ------------------------------------------------------------------

    def record_trade_opened(self, when: datetime | None = None) -> None:
        self.trades_today += 1
        self.last_trade_time = when or datetime.now(UTC)

    def record_trade_result(self, result: str) -> None:
        """Updates consecutive-loss counters from a closed trade result."""
        result = str(result).upper()
        if result == "LOSS":
            self.consec_losses_day += 1
            self.consec_losses_week += 1
        elif result == "WIN":
            self.consec_losses_day = 0
            self.consec_losses_week = 0
        # BE leaves the streak untouched

    def sync_counters_from_journal(self) -> None:
        """
        Rebuilds trades_today, consecutive-loss counters, and last_trade_time
        from trade_journal.py (source of truth in live mode).
        """
        try:
            import trade_journal

            closed = trade_journal.get_closed_trades()
            entries = trade_journal.get_entry_times()
        except Exception as exc:
            logger.warning("sync_counters_from_journal failed: %s", exc)
            return

        tz = _server_tz()
        now = server_now()
        today = now.date()
        iso_week = now.isocalendar()[:2]

        self.trades_today = sum(
            1 for ts in entries if ts.astimezone(tz).date() == today
        )
        self.last_trade_time = max(entries, default=None)

        def _trailing_losses(results: list[str]) -> int:
            count = 0
            for result in reversed(results):
                if result == "LOSS":
                    count += 1
                elif result == "WIN":
                    break
                # BE: keep looking back
            return count

        day_results = [
            r for ts, r in closed if ts.astimezone(tz).date() == today
        ]
        week_results = [
            r for ts, r in closed if ts.astimezone(tz).isocalendar()[:2] == iso_week
        ]
        self.consec_losses_day = _trailing_losses(day_results)
        self.consec_losses_week = _trailing_losses(week_results)

    # ------------------------------------------------------------------
    # Heartbeat (30-day inactivity closure protection)
    # ------------------------------------------------------------------

    def heartbeat_active(self) -> bool:
        """True when no trade has been opened in HEARTBEAT_DAYS days."""
        reference = self.last_trade_time or self.created_at
        idle_days = (datetime.now(UTC) - reference.astimezone(UTC)).days
        return idle_days >= config.HEARTBEAT_DAYS

    def risk_pct_for_next_trade(self) -> float:
        return config.HEARTBEAT_RISK_PCT if self.heartbeat_active() else config.RISK_PER_TRADE_PCT

    def adx_min_for_next_trade(self) -> float:
        return config.ADX_MIN_RELAXED if self.heartbeat_active() else config.ADX_MIN

    # ------------------------------------------------------------------
    # Drawdown checks
    # ------------------------------------------------------------------

    def current_drawdown_pct(self, current_equity: float) -> float:
        """Fractional drawdown from starting balance (0.0 when in profit)."""
        if self.starting_balance <= 0:
            return 0.0
        return max(
            (self.starting_balance - current_equity) / self.starting_balance, 0.0
        )

    def _drawdown_size_multiplier(self, current_equity: float) -> float:
        """
        Circuit breaker sizing factor:
          - full size below the warning threshold,
          - half size once drawdown >= DRAWDOWN_WARNING_THRESHOLD (2.0%),
          - zero once drawdown >= KILL_SWITCH_PCT (3.0%).
        The soft reduction never masks the hard stop: can_trade() disables
        trading entirely at the kill switch.
        """
        dd_pct = self.current_drawdown_pct(current_equity)
        if dd_pct >= config.KILL_SWITCH_PCT:
            return 0.0
        if dd_pct >= config.DRAWDOWN_WARNING_THRESHOLD:
            return 0.5
        return 1.0

    def check_absolute_drawdown(self, current_equity: float) -> bool:
        """
        Official Bootcamp backstop: returns False if absolute drawdown from
        starting_balance >= MAX_DRAWDOWN_LIMIT (−5% static). The kill switch
        at −3% should always fire first; this is belt-and-braces only.
        """
        if self.starting_balance <= 0:
            return True

        dd_pct = self.current_drawdown_pct(current_equity)

        if dd_pct >= config.DRAWDOWN_WARNING_THRESHOLD:
            logger.warning(
                f"Drawdown WARNING: {dd_pct:.2%} — position size reduced 50% "
                f"(kill switch {config.KILL_SWITCH_PCT:.2%}, "
                f"official limit {config.MAX_DRAWDOWN_LIMIT:.2%})"
            )

        if dd_pct >= config.MAX_DRAWDOWN_LIMIT:
            logger.critical(
                f"OFFICIAL MAX DRAWDOWN HIT ({dd_pct:.2%}). Emergency closing all positions."
            )
            # Import here to avoid circular dependency at module level
            from execution import close_all_positions
            close_all_positions()
            return False

        return True

    def check_daily_loss(self, current_equity: float) -> bool:
        """
        Auto-resets at server midnight, then returns False if the daily loss
        >= DAILY_LOSS_PCT (−0.75%). Sets self.halted_today = True on breach.
        """
        self._maybe_reset_daily(current_equity)

        if self.daily_start_balance <= 0:
            return True

        if self.halted_today:
            logger.warning("Trading already halted for today (daily loss limit).")
            return False

        loss_pct = (self.daily_start_balance - current_equity) / self.daily_start_balance

        if loss_pct >= config.DAILY_LOSS_PCT:
            logger.critical(
                f"DAILY LOSS LIMIT HIT ({loss_pct:.2%}). "
                f"Halting trading for the rest of the server day."
            )
            self.halted_today = True
            return False

        return True

    def check_weekly_loss(self, current_equity: float) -> bool:
        """
        Returns False if the weekly loss >= WEEKLY_STOP_PCT (−1.5% from
        week-start balance) or consecutive losses this week reach the cap.
        Halts until the next Monday (server time).
        """
        self._maybe_reset_weekly(current_equity)

        if self.halted_this_week:
            logger.warning("Trading already halted for this week.")
            return False

        if self.consec_losses_week >= config.MAX_CONSEC_LOSSES_WEEK:
            logger.critical(
                f"WEEKLY CONSECUTIVE LOSS LIMIT HIT ({self.consec_losses_week}). "
                f"Halting trading until Monday (server time)."
            )
            self.halted_this_week = True
            return False

        if self.week_start_balance <= 0:
            return True

        loss_pct = (self.week_start_balance - current_equity) / self.week_start_balance
        if loss_pct >= config.WEEKLY_STOP_PCT:
            logger.critical(
                f"WEEKLY LOSS LIMIT HIT ({loss_pct:.2%}). "
                f"Halting trading until Monday (server time)."
            )
            self.halted_this_week = True
            return False

        return True

    def check_daily_pacing(self) -> tuple[bool, str]:
        """Daily trade-count and consecutive-loss caps."""
        if self.trades_today >= config.MAX_TRADES_PER_DAY:
            return False, "MAX TRADES PER DAY"
        if self.consec_losses_day >= config.MAX_CONSEC_LOSSES_DAY:
            self.halted_today = True
            return False, "DAILY CONSECUTIVE LOSSES"
        return True, "OK"

    # ------------------------------------------------------------------
    # Sizing / SL-TP
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        equity: float,
        sl_price: float,
        entry_price: float,
        symbol: str,
        risk_pct: float | None = None,
    ) -> float:
        """
        Risk-based position sizing.

        risk_amount = equity * risk_pct (default RISK_PER_TRADE_PCT)
        sl_pips     = abs(entry - sl) / symbol pip size
        pip_value   = symbol pip value per standard lot
        lot         = risk_amount / (sl_pips * pip_value)

        Floored DOWN to the broker lot step (0.01) and clamped to max lot.
        Returns 0.0 if the calculated size is invalid or below min lot.
        Martingale prevention: never exceed the lot size used before a loss.
        """
        spec = get_symbol_spec(symbol)
        if spec is None:
            logger.error(f"calculate_lot_size: unsupported symbol {symbol}, returning 0.")
            return 0.0

        if equity <= 0 or entry_price == sl_price:
            logger.error("calculate_lot_size: invalid equity or SL distance, returning 0.")
            return 0.0

        if risk_pct is None:
            risk_pct = self.risk_pct_for_next_trade()
        risk_amount = equity * risk_pct
        sl_pips = abs(entry_price - sl_price) / spec.pip_size
        pip_value = spec.pip_value_per_standard_lot

        if risk_amount <= 0 or sl_pips <= 0 or pip_value <= 0:
            logger.error("calculate_lot_size: invalid risk inputs, returning 0.")
            return 0.0

        lot = risk_amount / (sl_pips * pip_value)

        # Drawdown circuit breaker:
        #   >= warning threshold → halve size; >= kill switch → no new position.
        size_mult = self._drawdown_size_multiplier(equity)
        if size_mult == 0.0:
            logger.critical(
                "Drawdown circuit breaker: kill-switch level reached — sizing "
                "blocked, returning 0."
            )
            return 0.0
        if size_mult < 1.0:
            logger.warning(
                "Drawdown circuit breaker: reducing position size to %.0f%%.",
                size_mult * 100,
            )
        lot *= size_mult

        if lot < spec.min_lot:
            logger.warning(
                f"calculate_lot_size: calculated lot {lot:.4f} below "
                f"minimum {spec.min_lot}, returning 0."
            )
            return 0.0

        lot = min(lot, spec.max_lot)
        lot = math.floor((lot / spec.lot_step) + 1e-12) * spec.lot_step
        lot = round(lot, 2)
        if lot < spec.min_lot:
            return 0.0

        # Martingale prevention:
        # If equity has declined since last trade, cap lot at previous lot size.
        if self._last_lot_size > 0 and equity < self._last_equity_at_lot:
            if lot > self._last_lot_size:
                logger.warning(
                    f"Martingale prevention: capping lot from {lot} to "
                    f"{self._last_lot_size} after equity decline."
                )
                lot = self._last_lot_size

        # Record for next trade
        self._last_lot_size = lot
        self._last_equity_at_lot = equity

        return lot

    def calculate_sl_tp(
        self,
        signal: int,
        entry_price: float,
        atr: float,
        swing_price: float | None = None,
        symbol: str | None = None,
    ) -> tuple[float, float] | None:
        """
        SL = SL_ATR_MULT (1.5) × ATR beyond the pullback swing (falls back to
        the entry price when no swing is supplied). The resulting SL distance
        is validated against [SL_MIN_PIPS, SL_MAX_PIPS]; outside the clamp the
        trade is SKIPPED (returns None) — the stop is never widened/narrowed.

        TP = TP_R (2.0) × the ACTUAL SL distance (R multiple, not ATR).

        Returns (sl, tp) rounded to 5 dp, or None to skip the trade.
        """
        return compute_sl_tp(signal, entry_price, atr, swing_price, symbol)

    @staticmethod
    def breakeven_trigger_price(signal: int, entry_price: float, sl_price: float) -> float:
        """Price at which SL should move to breakeven (+BE_AT_R × R)."""
        risk = abs(entry_price - sl_price)
        if signal == 1:
            return round(entry_price + config.BE_AT_R * risk, 5)
        return round(entry_price - config.BE_AT_R * risk, 5)

    # ------------------------------------------------------------------
    # Master gate
    # ------------------------------------------------------------------

    def can_trade(
        self, current_equity: float, open_trades: int
    ) -> tuple[bool, str]:
        """
        Master gate. Checks in priority order:
          a. Persisted disabled flag (kill switch, survives restart)
          b. Kill switch (−3% incl. floating → flatten + disable)
          c. Official −5% backstop
          d. Weekly stop (−1.5% or 5 consecutive losses)
          e. Daily loss (−0.75%, server-midnight auto-reset)
          f. Daily pacing (2 trades or 2 consecutive losses)
          g. Rollover / server no-trade window
          h. Max open trades
        Returns (True, "OK") only when all checks pass.
        """
        self._maybe_reset_daily(current_equity)

        if self.is_disabled():
            return False, "TRADING DISABLED (kill switch flag present)"

        if not self.check_kill_switch(current_equity):
            return False, "KILL SWITCH HIT"

        if not self.check_absolute_drawdown(current_equity):
            return False, "MAX DRAWDOWN HIT"

        if not self.check_weekly_loss(current_equity):
            return False, "WEEKLY STOP HIT"

        if not self.check_daily_loss(current_equity):
            return False, "DAILY LOSS LIMIT HIT"

        pacing_ok, pacing_reason = self.check_daily_pacing()
        if not pacing_ok:
            return False, pacing_reason

        if is_rollover_window() or is_no_trade_server_window():
            return False, "ROLLOVER WINDOW"

        if open_trades >= config.MAX_OPEN_TRADES:
            return False, "MAX TRADES OPEN"

        return True, "OK"
