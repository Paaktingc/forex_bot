"""
test_risk_manager.py

Unit tests for the risk_manager.py module.
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import risk_manager
import config

def test_calculate_position_size():
    equity = 10000
    # Risk = 10000 * 0.0075 = $75
    entry = 1.1000
    sl = 1.0950 # 50 pips / 0.0050 difference
    # risk_per_unit = 0.0050
    # units = 75 / 0.0050 = 15000 units
    # lots = 15000 / 100000 = 0.15 lots
    
    lots = risk_manager.calculate_position_size(equity, entry, sl)
    assert lots == 0.15
    
    # Zero risk test
    assert risk_manager.calculate_position_size(equity, entry, entry) == 0.0

def test_check_daily_loss_limit():
    initial_equity = 10000
    # 4% of 10000 is 400. Loss limit at 9600.
    
    # Allowed
    assert risk_manager.check_daily_loss_limit(initial_equity, 9700) == True
    
    # Not allowed (hit exactly 4%)
    assert risk_manager.check_daily_loss_limit(initial_equity, 9600) == False
    
    # Not allowed (exceeded 4%)
    assert risk_manager.check_daily_loss_limit(initial_equity, 9500) == False

def test_check_absolute_drawdown():
    starting_balance = 10000
    # 4.5% of 10000 is 450. Limit at 9550.
    
    assert risk_manager.check_absolute_drawdown(starting_balance, 9600) == True
    assert risk_manager.check_absolute_drawdown(starting_balance, 9550) == False
    assert risk_manager.check_absolute_drawdown(starting_balance, 9000) == False

def test_check_rollover_window():
    # Rollover is 21:00 to 22:00
    allowed_time = datetime(2023, 1, 1, 20, 30)
    blocked_time = datetime(2023, 1, 1, 21, 30)
    edge_case = datetime(2023, 1, 1, 22, 0)
    
    assert risk_manager.check_rollover_window(allowed_time) == True
    assert risk_manager.check_rollover_window(blocked_time) == False
    assert risk_manager.check_rollover_window(edge_case) == True

def test_check_max_concurrent_trades():
    assert risk_manager.check_max_concurrent_trades(1) == True
    assert risk_manager.check_max_concurrent_trades(2) == False
    assert risk_manager.check_max_concurrent_trades(3) == False

def test_check_order_delay():
    last_time = datetime(2023, 1, 1, 12, 0, 0)
    
    # 1 second later
    assert risk_manager.check_order_delay(last_time, last_time + timedelta(seconds=1)) == False
    
    # 3 seconds later
    assert risk_manager.check_order_delay(last_time, last_time + timedelta(seconds=3)) == True
