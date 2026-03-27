"""
risk_manager.py

Enforces daily loss limit (4%), absolute drawdown limit (4.5% of starting balance),
risk per trade (0.75%), maximum concurrent trades (2), rollover avoidance, 
and minimum delay between entries.
"""

import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)

def calculate_position_size(equity: float, entry_price: float, sl_price: float, contract_size: int = 100000) -> float:
    """
    Calculates the position size (in lots) based on the risk parameter.
    
    Args:
        equity (float): Current account equity.
        entry_price (float): The entry price of the trade.
        sl_price (float): The stop loss price.
        contract_size (int): Standard forex lot size (100,000 for standard lots).
        
    Returns:
        float: Position size in lots.
    """
    try:
        if entry_price == sl_price:
            return 0.0
            
        risk_amount = equity * config.RISK_PER_TRADE_PCT
        # Risk per unit is the pip/point distance in quote currency
        risk_per_unit = abs(entry_price - sl_price)
        
        # Position size in terms of base currency units
        position_units = risk_amount / risk_per_unit
        
        # Convert to lots
        lots = position_units / contract_size
        
        # typically round to 2 decimals for MT5 standard accounts
        lots = round(lots, 2)
        
        return lots
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0.0

def check_daily_loss_limit(initial_daily_equity: float, current_equity: float) -> bool:
    """
    Checks if the daily loss kill switch (4%) has been hit.
    
    Args:
        initial_daily_equity (float): Equity at the start of the trading day.
        current_equity (float): Current equity.
        
    Returns:
        bool: True if trading is ALLOWED, False if limit is hit.
    """
    if initial_daily_equity <= 0:
        return True
        
    loss_pct = (initial_daily_equity - current_equity) / initial_daily_equity
    if loss_pct >= config.DAILY_LOSS_KILL_SWITCH_PCT:
        logger.warning(f"DAILY LOSS LIMIT HIT. Loss: {loss_pct:.2%}, Limit: {config.DAILY_LOSS_KILL_SWITCH_PCT:.2%}")
        return False
    return True

def check_absolute_drawdown(starting_account_balance: float, current_equity: float) -> bool:
    """
    Checks if the absolute drawdown kill switch (4.5% from starting balance) has been hit.
    
    Args:
        starting_account_balance (float): Account balance at the very beginning of the phase.
        current_equity (float): Current equity.
        
    Returns:
        bool: True if trading is ALLOWED, False if limit is hit.
    """
    if starting_account_balance <= 0:
        return True
        
    drawdown_pct = (starting_account_balance - current_equity) / starting_account_balance
    if drawdown_pct >= config.DRAWDOWN_KILL_SWITCH_PCT:
        logger.warning(f"ABSOLUTE DRAWDOWN LIMIT HIT. DD: {drawdown_pct:.2%}, Limit: {config.DRAWDOWN_KILL_SWITCH_PCT:.2%}")
        return False
    return True

def check_rollover_window(current_time_utc: datetime) -> bool:
    """
    Checks if trading is blocked during the rollover window (21:00-22:00 UTC).
    
    Args:
        current_time_utc (datetime): Current UTC time.
        
    Returns:
        bool: True if trading is ALLOWED, False if blocked by rollover.
    """
    time_str = current_time_utc.strftime("%H:%M")
    if config.ROLLOVER_START_UTC <= time_str < config.ROLLOVER_END_UTC:
        logger.warning(f"Trading blocked during rollover window: {time_str}")
        return False
    return True

def check_max_concurrent_trades(open_trades_count: int) -> bool:
    """
    Checks if opening a new trade would exceed the maximum concurrent trades limit.
    
    Args:
        open_trades_count (int): Number of currently open trades.
        
    Returns:
        bool: True if trading is ALLOWED, False if limit is hit.
    """
    if open_trades_count >= config.MAX_CONCURRENT_TRADES:
        logger.warning(f"Max concurrent trades limit hit ({open_trades_count}/{config.MAX_CONCURRENT_TRADES})")
        return False
    return True

def check_order_delay(last_order_time: datetime, current_time: datetime) -> bool:
    """
    Checks if sufficient time (2s) has passed since the last order to prevent bulk ordering.
    
    Args:
        last_order_time (datetime): Time the last order was placed.
        current_time (datetime): Current time.
        
    Returns:
        bool: True if ALLOWED, False if delay rule applies.
    """
    if last_order_time is None:
        return True
        
    delta = (current_time - last_order_time).total_seconds()
    if delta < config.ORDER_DELAY_SECONDS:
        logger.warning(f"Order delay limit hit. Only {delta:.1f}s passed out of {config.ORDER_DELAY_SECONDS}s.")
        return False
    return True
