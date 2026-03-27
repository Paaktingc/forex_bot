import pytest
import os
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

import config
from news_filter import (
    _parse_ff_html,
    fetch_forex_factory_calendar,
    is_news_window,
    get_next_news_event,
    is_rollover_window,
    CACHE_FILE
)

MOCK_HTML = """
<html>
<body>
    <table class="calendar__table">
        <tr class="calendar__row">
            <td class="calendar__date">Mon Sep 25</td>
            <td class="calendar__time">10:30am</td>
            <td class="impact"><span class="icon--ff-impact-red"></span></td>
            <td class="currency">EUR</td>
            <td class="event">ECB President Lagarde Speaks</td>
        </tr>
        <tr class="calendar__row">
            <td class="calendar__date"></td>
            <td class="calendar__time">10:45am</td>
            <td class="impact"><span class="icon--ff-impact-yel" title="Low Impact Expected"></span></td>
            <td class="currency">USD</td>
            <td class="event">Minor Economic Data</td>
        </tr>
    </table>
</body>
</html>
"""

@pytest.fixture(autouse=True)
def clean_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    yield
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

def test_parse_ff_html():
    df = _parse_ff_html(MOCK_HTML)
    assert len(df) == 1
    assert df.loc[0, 'currency'] == 'EUR'
    assert df.loc[0, 'event'] == 'ECB President Lagarde Speaks'
    assert isinstance(df.loc[0, 'datetime_utc'], datetime)

@patch('requests.get')
def test_fetch_forex_factory_calendar_creates_cache(mock_get):
    mock_res = MagicMock()
    mock_res.text = MOCK_HTML
    mock_res.raise_for_status.return_value = None
    mock_get.return_value = mock_res
    
    assert not os.path.exists(CACHE_FILE)
    df = fetch_forex_factory_calendar()
    assert len(df) == 1
    assert os.path.exists(CACHE_FILE)
    
    # Second call should use cache (requests.get not called again)
    mock_get.reset_mock()
    df2 = fetch_forex_factory_calendar()
    mock_get.assert_not_called()
    assert len(df2) == 1

@patch('requests.get')
def test_is_news_window_scraping_fails_blocks_trading(mock_get):
    mock_get.side_effect = Exception("Connection error")
    
    # Should block trading (fail closed) => returns True
    assert is_news_window("EURUSD", 30) == True

@patch('news_filter.fetch_forex_factory_calendar')
def test_is_news_window(mock_fetch, monkeypatch):
    # Mock current time to 10:45 UTC
    now = datetime(2023, 9, 25, 10, 45, tzinfo=timezone.utc)
    
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return now
            
    monkeypatch.setattr('news_filter.datetime', MockDatetime)
    
    # News at 10:30 UTC
    df = pd.DataFrame([{
        'datetime_utc': datetime(2023, 9, 25, 10, 30, tzinfo=timezone.utc),
        'currency': 'EUR',
        'event': 'ECB Speaks'
    }])
    mock_fetch.return_value = df
    
    # 10:45 is within 30 mins of 10:30
    assert is_news_window("EURUSD", 30) == True
    
    # GBPUSD -> should not be affected by EUR news
    assert is_news_window("GBPUSD", 30) == False
    
    # 10:45 is outside 10 mins of 10:30
    assert is_news_window("EURUSD", 10) == False

@patch('news_filter.fetch_forex_factory_calendar')
def test_get_next_news_event(mock_fetch, monkeypatch):
    now = datetime(2023, 9, 25, 10, 0, tzinfo=timezone.utc)
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return now
    monkeypatch.setattr('news_filter.datetime', MockDatetime)
    
    df = pd.DataFrame([
        {
            'datetime_utc': datetime(2023, 9, 25, 11, 0, tzinfo=timezone.utc), # 60 mins away
            'currency': 'EUR',
            'event': 'ECB Speaks'
        },
        {
            'datetime_utc': datetime(2023, 9, 25, 9, 0, tzinfo=timezone.utc), # Past event
            'currency': 'USD',
            'event': 'Old News'
        }
    ])
    mock_fetch.return_value = df
    
    nxt = get_next_news_event("EURUSD")
    assert nxt is not None
    assert nxt['currency'] == 'EUR'
    assert nxt['minutes_until'] == 60
    assert nxt['event_name'] == 'ECB Speaks'

def test_is_rollover_window(monkeypatch):
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2023, 9, 25, config.ROLLOVER_START_UTC, 30, tzinfo=timezone.utc)
    monkeypatch.setattr('news_filter.datetime', MockDatetime)
    
    assert is_rollover_window() == True
    
    class MockDatetimeFalse:
        @classmethod
        def now(cls, tz=None):
            return datetime(2023, 9, 25, 12, 30, tzinfo=timezone.utc)
    monkeypatch.setattr('news_filter.datetime', MockDatetimeFalse)
    
    assert is_rollover_window() == False
