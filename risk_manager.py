"""
risk_manager.py

Enforces The5ers hard risk rules via RiskManager class:
  - Absolute drawdown kill-switch  (4.5% from starting balance)
  - Daily loss kill-switch          (4.0% from daily start balance)
  - Position sizing                 (ATR-based, 0.75% risk per trade)
  - SL/TP calculation               (ATR multipliers from config)
  - can_trade() gate                (combines all checks)
"""

import logging
import math
from datetime import datetime, timezone

import config
from symbol_specs import get_symbol_spec

logger = logging.getLogger(__name__)

UTC = timezone.utc


def is_rollover_window() -> bool:
    """
    Returns True if the current UTC time falls inside the rollover window
    (ROLLOVER_START_UTC <= hour < ROLLOVER_END_UTC).
    """
    now_hour = datetime.now(UTC).hour
    return config.ROLLOVER_START_UTC <= now_hour < config.ROLLOVER_END_UTC


class RiskManager:
    """Stateful risk manager for the The5ers forex bot."""

    def __init__(self, starting_balance: float) -> None:
        self.starting_balance: float = starting_balance
        self.daily_start_balance: float = starting_balance
        self.daily_start_time: datetime = datetime.now(UTC)
        self.halted_today: bool = False

        # Martingale prevention: track last lot size
        self._last_lot_size: float = 0.0
        self._last_equity_at_lot: float = starting_balance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self, current_equity: float) -> None:
        """
        Resets daily_start_balance and halted_today when a new UTC day starts.
        """
        now = datetime.now(UTC)
        if now.date() > self.daily_start_time.date():
            logger.info(
                f"New UTC day detected. Resetting daily balance from "
                f"{self.daily_start_balance} to current equity {current_equity}."
            )
            self.daily_start_time = now
            self.daily_start_balance = current_equity
            self.halted_today = False

    # ------------------------------------------------------------------
    # Public methods
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
          - half size once drawdown >= DRAWDOWN_WARNING_THRESHOLD (3.0%),
          - zero (halt) once drawdown >= MAX_DRAWDOWN_LIMIT (4.5%).
        """
        dd_pct = self.current_drawdown_pct(current_equity)
        if dd_pct >= config.MAX_DRAWDOWN_LIMIT:
            return 0.0
        if dd_pct >= config.DRAWDOWN_WARNING_THRESHOLD:
            return 0.5
        return 1.0

    def check_absolute_drawdown(self, current_equity: float) -> bool:
        """
        Returns False (stop trading) if absolute drawdown from starting_balance
        >= MAX_DRAWDOWN_PCT.

        Logs WARNING at 3 %, CRITICAL at 4 %, and calls
        execution.close_all_positions() on breach.
        """
        if self.starting_balance <= 0:
            return True

        dd_pct = self.current_drawdown_pct(current_equity)

        if dd_pct >= config.DRAWDOWN_WARNING_THRESHOLD:
            logger.warning(
                f"Drawdown WARNING: {dd_pct:.2%} — position size reduced 50% "
                f"(hard limit {config.MAX_DRAWDOWN_LIMIT:.2%})"
            )

        if dd_pct >= config.MAX_DRAWDOWN_LIMIT:
            logger.critical(
                f"MAX DRAWDOWN HIT ({dd_pct:.2%}). Emergency closing all positions."
            )
            # Import here to avoid circular dependency at module level
            from execution import close_all_positions
            close_all_positions()
            return False

        return True

    def check_daily_loss(self, current_equity: float) -> bool:
        """
        Auto-resets at UTC midnight, then returns False if the daily loss
        >= DAILY_LOSS_PCT. Sets self.halted_today = True on breach.
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
                f"Halting trading for the rest of the UTC day."
            )
            self.halted_today = True
            return False

        return True

    def calculate_lot_size(
        self,
        equity: float,
        sl_price: float,
        entry_price: float,
        symbol: str,
    ) -> float:
        """
        Risk-based position sizing.

        risk_amount = equity * RISK_PER_TRADE_PCT
        sl_pips     = abs(entry - sl) / symbol pip size
        pip_value   = symbol pip value per standard lot
        lot         = risk_amount / (sl_pips * pip_value)

        Floored to the broker lot step and clamped to max lot.
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

        risk_amount = equity * config.RISK_PER_TRADE_PCT
        sl_pips = abs(entry_price - sl_price) / spec.pip_size
        pip_value = spec.pip_value_per_standard_lot

        if risk_amount <= 0 or sl_pips <= 0 or pip_value <= 0:
            logger.error("calculate_lot_size: invalid risk inputs, returning 0.")
            return 0.0

        lot = risk_amount / (sl_pips * pip_value)

        # Drawdown circuit breaker (The5ers protection):
        #   >= warning threshold → halve size; >= hard limit → no new position.
        size_mult = self._drawdown_size_multiplier(equity)
        if size_mult == 0.0:
            logger.critical(
                "Drawdown circuit breaker: hard limit reached — sizing blocked, "
                "returning 0."
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
        self, signal: int, entry_price: float, atr: float
    ) -> tuple[float, float]:
        """
        ATR-based SL/TP.
        signal == 1  → Buy:  sl below entry, tp above entry
        signal == -1 → Sell: sl above entry, tp below entry

        Returns (sl, tp) rounded to 5 dp.
        """
        if signal == 1:   # Buy
            sl = entry_price - config.SL_ATR_MULT * atr
            tp = entry_price + config.TP_ATR_MULT * atr
        else:             # Sell
            sl = entry_price + config.SL_ATR_MULT * atr
            tp = entry_price - config.TP_ATR_MULT * atr

        return round(sl, 5), round(tp, 5)

    def can_trade(
        self, current_equity: float, open_trades: int
    ) -> tuple[bool, str]:
        """
        Master gate. Checks in priority order:
          a. Absolute drawdown
          b. Daily loss (with midnight auto-reset)
          c. Rollover window
          d. Max open trades
        Returns (True, "OK") only when all checks pass.
        """
        self._maybe_reset_daily(current_equity)

        if not self.check_absolute_drawdown(current_equity):
            return False, "MAX DRAWDOWN HIT"

        if not self.check_daily_loss(current_equity):
            return False, "DAILY LOSS LIMIT HIT"

        if is_rollover_window():
            return False, "ROLLOVER WINDOW"

        if open_trades >= config.MAX_OPEN_TRADES:
            return False, "MAX TRADES OPEN"

        return True, "OK"
