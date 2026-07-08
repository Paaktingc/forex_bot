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
    
    # Non-major high-impact news at 10:30 UTC
    df = pd.DataFrame([{
        'datetime_utc': datetime(2023, 9, 25, 10, 30, tzinfo=timezone.utc),
        'currency': 'EUR',
        'event': 'German Ifo Business Climate'
    }])
    mock_fetch.return_value = df

    # 10:45 is within 30 mins of 10:30
    assert is_news_window("EURUSD", 30) == True

    # GBPUSD -> should not be affected by EUR news... but with zero relevant
    # events remaining the frame is non-empty, so trading is allowed
    assert is_news_window("GBPUSD", 30) == False

    # 10:45 is outside 10 mins of 10:30 (non-major event honours buffer_mins)
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


# ---------------------------------------------------------------------------
# Tiered windows, flatten rule, and fail-closed behaviour
# ---------------------------------------------------------------------------

from news_filter import is_major_event, should_flatten_for_news


def _mock_now(monkeypatch, now):
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return now
    monkeypatch.setattr('news_filter.datetime', MockDatetime)


def test_is_major_event_classification():
    assert is_major_event("Non-Farm Employment Change", "USD") is True
    assert is_major_event("CPI m/m", "USD") is True
    assert is_major_event("FOMC Statement", "USD") is True
    assert is_major_event("Main Refinancing Rate", "EUR") is True
    assert is_major_event("ECB Press Conference", "EUR") is True
    assert is_major_event("German Ifo Business Climate", "EUR") is False
    # CPI for a non-EUR/USD currency is not "US CPI"
    assert is_major_event("CPI y/y", "GBP") is False


@patch('news_filter.fetch_forex_factory_calendar')
def test_major_event_gets_60_min_window(mock_fetch, monkeypatch):
    now = datetime(2023, 9, 25, 10, 0, tzinfo=timezone.utc)
    _mock_now(monkeypatch, now)

    # NFP 50 minutes ahead: outside the 30-min window, inside the 60-min one
    df = pd.DataFrame([{
        'datetime_utc': datetime(2023, 9, 25, 10, 50, tzinfo=timezone.utc),
        'currency': 'USD',
        'event': 'Non-Farm Employment Change'
    }])
    mock_fetch.return_value = df
    assert is_news_window("EURUSD") == True

    # The same distance for a non-major event is NOT blocked
    df_minor = df.copy()
    df_minor.loc[0, 'event'] = 'Existing Home Sales'
    mock_fetch.return_value = df_minor
    assert is_news_window("EURUSD") == False


@patch('news_filter.fetch_forex_factory_calendar')
def test_empty_calendar_fails_closed(mock_fetch):
    mock_fetch.return_value = pd.DataFrame()
    assert is_news_window("EURUSD") == True
    assert should_flatten_for_news("EURUSD") == True


@patch('requests.get')
def test_should_flatten_fails_closed_on_fetch_error(mock_get):
    mock_get.side_effect = Exception("Connection error")
    assert should_flatten_for_news("EURUSD") == True


@patch('news_filter.fetch_forex_factory_calendar')
def test_flatten_15_min_before_major(mock_fetch, monkeypatch):
    now = datetime(2023, 9, 25, 10, 0, tzinfo=timezone.utc)
    _mock_now(monkeypatch, now)

    def _frame(event, minutes_ahead, currency='USD'):
        return pd.DataFrame([{
            'datetime_utc': now + timedelta(minutes=minutes_ahead),
            'currency': currency,
            'event': event,
        }])

    # FOMC in 10 minutes → flatten
    mock_fetch.return_value = _frame('FOMC Statement', 10)
    assert should_flatten_for_news("EURUSD") == True

    # FOMC in 30 minutes → hold (entries already blocked by the ±60 window)
    mock_fetch.return_value = _frame('FOMC Statement', 30)
    assert should_flatten_for_news("EURUSD") == False

    # FOMC 10 minutes AGO → no flatten (event already released)
    mock_fetch.return_value = _frame('FOMC Statement', -10)
    assert should_flatten_for_news("EURUSD") == False

    # Non-major event in 10 minutes → no flatten
    mock_fetch.return_value = _frame('Existing Home Sales', 10)
    assert should_flatten_for_news("EURUSD") == False
