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
    sl_atr_mult: float | None = None,
    tp_r: float | None = None,
) -> tuple[float, float] | None:
    """
    Stateless SL/TP math (shared by RiskManager and the backtester).

    SL = SL_ATR_MULT (1.5) × ATR beyond the pullback swing (falls back to the
    entry price when no swing is supplied). The resulting SL distance must be
    inside [SL_MIN_PIPS, SL_MAX_PIPS]; otherwise the trade is SKIPPED
    (returns None) — the stop is never widened or narrowed to fit.

    TP = TP_R (2.0) × the ACTUAL SL distance (an R multiple, not ATR).

    sl_atr_mult / tp_r default to config; overrides exist for research
    (walk-forward perturbation and exit sweeps) only.
    """
    if atr is None or atr <= 0:
        return None
    if sl_atr_mult is None:
        sl_atr_mult = config.SL_ATR_MULT
    if tp_r is None:
        tp_r = config.TP_R

    spec = get_symbol_spec(symbol or config.SYMBOL)
    pip_size = spec.pip_size if spec else 0.0001

    anchor = swing_price if swing_price is not None else entry_price
    if signal == 1:   # Buy: SL below the pullback swing low
        sl = anchor - sl_atr_mult * atr
        sl_distance = entry_price - sl
    else:             # Sell: SL above the pullback swing high
        sl = anchor + sl_atr_mult * atr
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
        tp = entry_price + tp_r * sl_distance
    else:
        tp = entry_price - tp_r * sl_distance

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

        # High Stakes "minimum profitable days" gate (no-op for Bootcamp, where
        # config.MIN_PROFITABLE_DAYS == 0). A profitable day = the day closes
        # with realized gain >= PROFITABLE_DAY_MIN_PCT of the step balance.
        self.profitable_days: int = 0

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
    # Persistent risk state (Finding 1: baseline must survive restart)
    # ------------------------------------------------------------------
    #
    # The STEP-START balance is the anchor for the kill switch and the official
    # max-loss backstop. It must NEVER be re-derived from the live broker
    # balance on restart, or a restart after losses silently moves the −3%/−6%
    # kill baseline downward. We persist it (plus daily/weekly baselines and
    # counters) to RISK_STATE_PATH and restore it verbatim on startup. Starting
    # a genuinely new step (fresh account balance) is a DELIBERATE operator
    # action via RiskManager.load_or_init(reset=True) — never inferred.

    STATE_VERSION = 1

    @property
    def state_path(self) -> Path:
        return Path(config.RISK_STATE_PATH)

    def to_state(self) -> dict:
        """Serializable snapshot of everything needed to resume safely."""
        return {
            "version": self.STATE_VERSION,
            "programme": getattr(config, "PROGRAMME", "bootcamp"),
            "starting_balance": self.starting_balance,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "daily_start_balance": self.daily_start_balance,
            "daily_start_time": self.daily_start_time.astimezone(UTC).isoformat(),
            "week_start_balance": self.week_start_balance,
            "week_start_time": self.week_start_time.astimezone(UTC).isoformat(),
            "halted_today": self.halted_today,
            "halted_this_week": self.halted_this_week,
            "trades_today": self.trades_today,
            "consec_losses_day": self.consec_losses_day,
            "consec_losses_week": self.consec_losses_week,
            "profitable_days": self.profitable_days,
            "last_trade_time": (
                self.last_trade_time.astimezone(UTC).isoformat()
                if self.last_trade_time else None
            ),
        }

    def save_state(self) -> None:
        """Atomically persist the risk state. Best-effort: never raises."""
        try:
            path = self.state_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.to_state(), indent=2))
            tmp.replace(path)
        except Exception as exc:  # persistence must not crash the trader
            logger.warning("save_state failed (%s) — continuing.", exc)

    def _apply_state(self, st: dict) -> None:
        """Restore baselines/counters from a persisted state dict."""
        self.starting_balance = float(st["starting_balance"])
        self.created_at = datetime.fromisoformat(st["created_at"])
        self.daily_start_balance = float(st["daily_start_balance"])
        self.daily_start_time = datetime.fromisoformat(st["daily_start_time"])
        self.week_start_balance = float(st["week_start_balance"])
        self.week_start_time = datetime.fromisoformat(st["week_start_time"])
        self.halted_today = bool(st.get("halted_today", False))
        self.halted_this_week = bool(st.get("halted_this_week", False))
        self.trades_today = int(st.get("trades_today", 0))
        self.consec_losses_day = int(st.get("consec_losses_day", 0))
        self.consec_losses_week = int(st.get("consec_losses_week", 0))
        self.profitable_days = int(st.get("profitable_days", 0))
        lt = st.get("last_trade_time")
        self.last_trade_time = datetime.fromisoformat(lt) if lt else None

    @classmethod
    def load_or_init(
        cls,
        broker_balance: float,
        reset: bool = False,
        disabled_flag_path: str | Path | None = None,
    ) -> "RiskManager":
        """
        Startup factory. Restores the persisted STEP-START baseline if a valid
        state file exists for the active programme; otherwise creates a fresh
        baseline from the live broker balance and persists it.

        reset=True forces a fresh baseline from broker_balance (use ONLY when
        deliberately starting a new challenge step / new account).
        """
        rm = cls(broker_balance, disabled_flag_path=disabled_flag_path)
        path = Path(config.RISK_STATE_PATH)

        if reset:
            logger.warning(
                "RiskManager.load_or_init(reset=True): re-baselining to broker "
                "balance %.2f and overwriting %s.", broker_balance, path,
            )
            rm.save_state()
            return rm

        if not path.exists():
            logger.info(
                "No persisted risk state at %s — initializing fresh baseline "
                "from broker balance %.2f.", path, broker_balance,
            )
            rm.save_state()
            return rm

        try:
            st = json.loads(path.read_text())
        except Exception as exc:
            logger.error(
                "Risk state at %s is unreadable (%s) — refusing to guess. "
                "Initializing fresh baseline from broker balance and overwriting.",
                path, exc,
            )
            rm.save_state()
            return rm

        active = getattr(config, "PROGRAMME", "bootcamp")
        if st.get("programme") != active:
            logger.warning(
                "Persisted risk state is for programme %r but active programme "
                "is %r — NOT reusing its baseline. Starting fresh (use "
                "--new-step if this is intentional).", st.get("programme"), active,
            )
            rm.save_state()
            return rm

        rm._apply_state(st)
        drift = broker_balance - rm.starting_balance
        logger.critical(
            "RiskManager RESTORED persisted baseline: step-start=%.2f "
            "(broker now=%.2f, drift=%+.2f). Kill/max-loss anchored to the "
            "persisted step-start, NOT the current balance.",
            rm.starting_balance, broker_balance, drift,
        )
        if broker_balance > rm.starting_balance * (1.0 + config.PROFIT_TARGET_PCT + 0.01):
            logger.critical(
                "Broker balance is well ABOVE the persisted step-start — this "
                "looks like a NEW step. If so, relaunch with --new-step to "
                "re-baseline. Continuing with the OLD baseline for safety.",
            )
        return rm

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
        equity INCLUDING floating P&L is down KILL_SWITCH_PCT from the initial
        step balance (−3% Bootcamp / −6% High Stakes, per the active profile).
        The official MAX_DRAWDOWN_LIMIT (−5% / −10%) must never be reached.
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
            # Credit a profitable day BEFORE resetting the day-start baseline.
            # The5ers defines it as: min(midnight balance, midnight equity) −
            # previous-day balance >= PROFITABLE_DAY_MIN_PCT of the balance.
            # We approximate with equity at the day boundary (the London bot is
            # flat overnight, so equity ≈ balance here).
            if config.MIN_PROFITABLE_DAYS and self.starting_balance > 0:
                day_profit = current_equity - self.daily_start_balance
                threshold = config.PROFITABLE_DAY_MIN_PCT * self.starting_balance
                if day_profit >= threshold:
                    self.profitable_days += 1
                    logger.info(
                        "Profitable day credited (%.2f%% of balance). "
                        "Profitable days: %d/%d.",
                        day_profit / self.starting_balance * 100.0,
                        self.profitable_days, config.MIN_PROFITABLE_DAYS,
                    )
            logger.info(
                f"New server day detected. Resetting daily balance from "
                f"{self.daily_start_balance} to current equity {current_equity}."
            )
            self.daily_start_time = now
            self.daily_start_balance = current_equity
            self.halted_today = False
            self.trades_today = 0
            self.consec_losses_day = 0
            self.save_state()  # new day baseline must survive a restart
        self._maybe_reset_weekly(current_equity, now)

    def profitable_days_met(self) -> bool:
        """
        True when the programme's minimum-profitable-days requirement is
        satisfied (always True for programmes that don't require any, e.g.
        Bootcamp). High Stakes requires >= config.MIN_PROFITABLE_DAYS before a
        step's profit target counts as a pass.
        """
        return self.profitable_days >= config.MIN_PROFITABLE_DAYS

    def check_official_daily_loss(self, current_equity: float) -> bool:
        """
        Official daily-loss backstop (High Stakes: 5% of day-start balance).
        Returns False and disables trading if breached. No-op for programmes
        without an official daily limit (Bootcamp steps → OFFICIAL_DAILY_LOSS_PCT
        is None). The self-imposed check_daily_loss() pacing stop (−1.5%) fires
        long before this; this is belt-and-braces against a fast gap day.
        """
        limit = config.OFFICIAL_DAILY_LOSS_PCT
        if not limit or self.daily_start_balance <= 0:
            return True
        loss_pct = max(
            (self.daily_start_balance - current_equity) / self.daily_start_balance,
            0.0,
        )
        if loss_pct >= limit:
            self.disable_trading(
                f"OFFICIAL DAILY LOSS: {loss_pct:.2%} >= {limit:.2%} of "
                f"day-start balance",
                current_equity=current_equity,
            )
            return False
        return True

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
        self.save_state()

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
        self.save_state()

    # ------------------------------------------------------------------
    # Evaluation-target stop (Finding 4)
    # ------------------------------------------------------------------

    def profit_target_reached(self, current_equity: float) -> bool:
        """
        True once equity (incl. floating P&L) has reached the active step's
        profit target AND the programme's minimum-profitable-days requirement
        is met (always satisfied for Bootcamp, which requires none).
        """
        if self.starting_balance <= 0:
            return False
        gain_pct = (current_equity - self.starting_balance) / self.starting_balance
        return gain_pct >= config.PROFIT_TARGET_PCT and self.profitable_days_met()

    def halt_for_target(self, current_equity: float) -> None:
        """
        Locks in a passing step: flattens open positions and persists a
        BENIGN halt flag so a restart cannot resume trading and give the pass
        back. Distinct from the kill switch — the reason string makes clear
        this is a SUCCESS. Operator re-arms for the next step by relaunching
        with --new-step (which re-baselines and clears the flag).
        """
        self.disable_trading(
            f"STEP TARGET REACHED (+{config.PROFIT_TARGET_PCT:.0%}) — step "
            f"passed. Do NOT keep trading. Start the next step with "
            f"--new-step. Equity {current_equity:.2f} vs step-start "
            f"{self.starting_balance:.2f}.",
            current_equity=current_equity,
        )

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
          a. Persisted disabled flag (kill switch / passed step, survives restart)
          b. Kill switch (−3%/−6% incl. floating → flatten + disable)
          c. Official max-loss backstop (−5%/−10%)
          d. Profit target reached → halt (lock in the pass; Finding 4)
          e. Official daily loss (High Stakes 5%; None for Bootcamp steps)
          f. Weekly stop (−1.5%/−3% or consecutive losses)
          g. Daily loss (self-imposed pacing, server-midnight auto-reset)
          h. Daily pacing (trade count / consecutive losses)
          i. Rollover / server no-trade window
          j. Max open trades
        Returns (True, "OK") only when all checks pass.
        """
        self._maybe_reset_daily(current_equity)

        if self.is_disabled():
            return False, "TRADING DISABLED (kill switch / passed-step flag present)"

        if not self.check_kill_switch(current_equity):
            return False, "KILL SWITCH HIT"

        if not self.check_absolute_drawdown(current_equity):
            return False, "MAX DRAWDOWN HIT"

        # Evaluation-target stop: once the step is passed, STOP. Trading on
        # would risk giving back a qualifying result (Finding 4).
        if self.profit_target_reached(current_equity):
            self.halt_for_target(current_equity)
            return False, "STEP TARGET REACHED — step passed, trading halted"

        # Programme-specific official backstop.  Bootcamp configures this as
        # None; High Stakes configures the firm's 5% daily loss limit.  Keep
        # this separate from the tighter self-imposed pacing stop below so a
        # future pacing change cannot silently remove the official rule.
        if not self.check_official_daily_loss(current_equity):
            return False, "OFFICIAL DAILY LOSS LIMIT HIT"

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
