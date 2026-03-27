"""
test_news_filter.py

Unit tests for the news_filter.py module.
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from news_filter import fetch_high_impact_news, is_trading_allowed_by_news
import config

def test_fetch_high_impact_news():
    now = datetime.utcnow()
    news = fetch_high_impact_news(now)
    assert isinstance(news, list)

def test_is_trading_allowed_by_news():
    news_time = datetime(2023, 1, 1, 13, 30)
    
    # 31 mins before - Allowed
    current_time_before = news_time - timedelta(minutes=31)
    assert is_trading_allowed_by_news(current_time_before, [news_time]) == True
    
    # 29 mins before - Blocked
    current_time_blocked_before = news_time - timedelta(minutes=29)
    assert is_trading_allowed_by_news(current_time_blocked_before, [news_time]) == False
    
    # Exactly on news time - Blocked
    assert is_trading_allowed_by_news(news_time, [news_time]) == False
    
    # 29 mins after - Blocked
    current_time_blocked_after = news_time + timedelta(minutes=29)
    assert is_trading_allowed_by_news(current_time_blocked_after, [news_time]) == False
    
    # 31 mins after - Allowed
    current_time_after = news_time + timedelta(minutes=31)
    assert is_trading_allowed_by_news(current_time_after, [news_time]) == True
