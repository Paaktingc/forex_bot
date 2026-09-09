"""
news_filter.py

Scrapes Forex Factory or similar economic calendar to find high-impact news times.

Tiered no-entry windows (The5ers Bootcamp safe policy):
  - any high-impact EUR/USD event:            ±NEWS_BUFFER_MINS (30 min)
  - MAJOR events (NFP, US CPI, FOMC, ECB):    ±NEWS_MAJOR_BUFFER_MINS (60 min)
    plus flatten open positions NEWS_FLATTEN_BEFORE_MINS (15 min) before.

FAIL CLOSED: if the calendar cannot be fetched OR fetches but parses to zero
events (layout change), trading is treated as inside a blackout.

Pending orders near news are never used: the bot places market orders only
(bracketing/straddles are prohibited — enforced in execution.py by design).
"""

import logging
import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import config

logger = logging.getLogger(__name__)


def _news_source_tz() -> ZoneInfo:
    """The IANA timezone Forex Factory renders event times in (config)."""
    try:
        return ZoneInfo(getattr(config, "NEWS_SOURCE_TZ", "UTC"))
    except Exception:
        logger.error(
            "Invalid NEWS_SOURCE_TZ=%r — falling back to UTC.",
            getattr(config, "NEWS_SOURCE_TZ", None),
        )
        return ZoneInfo("UTC")

CACHE_FILE = config.DATA_DIR / "news_cache.json"
CACHE_EXPIRY_HOURS = 6
FF_CALENDAR_URL = "https://www.forexfactory.com/calendar"

# Events that get the wider ±60 min window and the flatten-15-min-before rule.
MAJOR_EVENT_KEYWORDS = (
    "non-farm",
    "nonfarm",
    "nfp",
    "cpi",
    "fomc",
    "federal funds rate",
    "ecb",
    "main refinancing rate",
)
MAJOR_EVENT_CURRENCIES = {"USD", "EUR"}


def is_major_event(event_name: str, currency: str = "") -> bool:
    """True for NFP, US CPI, FOMC, and ECB rate decisions/pressers."""
    name = str(event_name).lower()
    if currency and str(currency).upper() not in MAJOR_EVENT_CURRENCIES:
        return False
    return any(keyword in name for keyword in MAJOR_EVENT_KEYWORDS)

def _parse_ff_html(html: str) -> pd.DataFrame:
    """Parses Forex Factory HTML and returns a DataFrame of high-impact events."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="calendar__table")
    if not table:
        return pd.DataFrame()
        
    events = []
    current_date_str = ""
    current_time_str = ""
    
    # Forex Factory parsing is tricky because date and time carry over across rows
    for row in table.find_all("tr", class_="calendar__row"):
        # Date
        date_elem = row.find("td", class_="calendar__date")
        if date_elem and date_elem.text.strip():
            # e.g., "Mon Sep 25" -> we will need to guess the year or simply 
            # assume current year for the sake of UTC mapping
            date_text = date_elem.text.strip()
            # Typically features Day abbreviated, e.g. "Mon Sep 25"
            # We strip the day of week to parse just "Sep 25"
            parts = date_text.split()
            if len(parts) >= 3:
                current_date_str = f"{parts[1]} {parts[2]}" 
        
        # Time
        time_elem = row.find("td", class_="calendar__time")
        if time_elem and time_elem.text.strip():
            text = time_elem.text.strip()
            # Skip all-day or tentative events if they don't have parseable time
            if "All Day" not in text and "Tentative" not in text:
                current_time_str = text
                
        # Impact
        impact_elem = row.find("td", class_="impact")
        is_high_impact = False
        if impact_elem:
            span = impact_elem.find("span")
            if span and ('icon--ff-impact-red' in span.get("class", []) or 'high' in span.get("title", "").lower()):
                is_high_impact = True
                
        # Curriculum
        currency_elem = row.find("td", class_="currency")
        currency = currency_elem.text.strip() if currency_elem else ""
        
        # Event Name
        event_elem = row.find("td", class_="event")
        event_name = event_elem.text.strip() if event_elem else ""
        
        if is_high_impact and current_date_str and current_time_str and currency:
            # Construct a naive datetime
            year = datetime.now(timezone.utc).year
            datetime_str = f"{year} {current_date_str} {current_time_str}"
            try:
                # e.g., "2023 Sep 25 10:30am"
                dt_naive = datetime.strptime(datetime_str, "%Y %b %d %I:%M%p")

                # Forex Factory renders times in the account/cookie timezone
                # (config.NEWS_SOURCE_TZ), NOT UTC. Localize to that zone, then
                # convert to UTC so downstream blackout math is correct
                # regardless of how the FF session is configured (Finding 3).
                dt_utc = (
                    dt_naive.replace(tzinfo=_news_source_tz())
                    .astimezone(timezone.utc)
                )

                events.append({
                    "datetime_utc": dt_utc,
                    "currency": currency,
                    "event": event_name
                })
            except ValueError:
                pass
                
    return pd.DataFrame(events)

def fetch_forex_factory_calendar() -> pd.DataFrame:
    """
    Scrapes the Forex Factory calendar for high-impact events.
    Caches the result locally for 6 hours.
    Returns:
        pd.DataFrame with columns: datetime_utc, currency, event.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached_data = json.load(f)
                
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now(timezone.utc) - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                logger.info("Using cached news events.")
                df = pd.DataFrame(cached_data["events"])
                if not df.empty:
                    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
                return df
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}. Refetching.")
            
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        res = requests.get(FF_CALENDAR_URL, headers=headers, timeout=15)
        res.raise_for_status()
        
        df = _parse_ff_html(res.text)
        
        # Save to cache
        cache_content = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "events": []
        }
        
        if not df.empty:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            # Convert to dict, making datetime JSON serializable
            events_list = df.copy()
            events_list['datetime_utc'] = events_list['datetime_utc'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            cache_content["events"] = events_list.to_dict(orient="records")
            
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_content, f)
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Forex Factory calendar: {e}")
        raise RuntimeError("News Scraping Failed")

def _load_calendar_fail_closed(symbol: str) -> pd.DataFrame | None:
    """
    Returns the relevant-currency event frame, or None to signal BLACKOUT
    (fetch failed or calendar parsed to zero events — fail closed).
    """
    try:
        df = fetch_forex_factory_calendar()
    except Exception:
        logger.error("News calendar unavailable. Failing closed (blocking trading).")
        return None

    if df is None or df.empty:
        logger.error(
            "News calendar fetched but contains ZERO events — treating as a "
            "parser/layout failure and failing closed (blocking trading)."
        )
        return None

    c1 = symbol[:3]
    c2 = symbol[3:6]
    return df[df["currency"].isin([c1, c2])]


def _minutes_from_event(row, now: datetime) -> float:
    dt_news = row["datetime_utc"]
    if dt_news.tzinfo is None:
        dt_news = dt_news.replace(tzinfo=timezone.utc)
    return (dt_news - now).total_seconds() / 60.0


def is_news_window(symbol: str, buffer_mins: int | None = None) -> bool:
    """
    True when trading should be BLOCKED:
      - inside ±buffer_mins (default NEWS_BUFFER_MINS) of any high-impact
        event for the symbol's currencies,
      - inside ±NEWS_MAJOR_BUFFER_MINS of a MAJOR event (NFP/CPI/FOMC/ECB),
      - or when the calendar is unavailable/empty (fail closed).
    """
    if buffer_mins is None:
        buffer_mins = config.NEWS_BUFFER_MINS

    df_sym = _load_calendar_fail_closed(symbol)
    if df_sym is None:
        return True

    now = datetime.now(timezone.utc)
    for _, row in df_sym.iterrows():
        window = (
            config.NEWS_MAJOR_BUFFER_MINS
            if is_major_event(row["event"], row["currency"])
            else buffer_mins
        )
        diff_mins = abs(_minutes_from_event(row, now))
        if diff_mins <= window:
            logger.warning(
                f"News block: {row['event']} within ±{window} min "
                f"({diff_mins:.1f} min away)"
            )
            return True

    return False


def should_flatten_for_news(symbol: str) -> bool:
    """
    True when open positions must be CLOSED because a MAJOR event
    (NFP/US CPI/FOMC/ECB) starts within NEWS_FLATTEN_BEFORE_MINS minutes.
    Fails closed: calendar unavailable ⇒ flatten.
    """
    df_sym = _load_calendar_fail_closed(symbol)
    if df_sym is None:
        return True

    now = datetime.now(timezone.utc)
    for _, row in df_sym.iterrows():
        if not is_major_event(row["event"], row["currency"]):
            continue
        minutes_until = _minutes_from_event(row, now)
        if 0 <= minutes_until <= config.NEWS_FLATTEN_BEFORE_MINS:
            logger.warning(
                f"MAJOR news flatten: {row['event']} in {minutes_until:.1f} min "
                f"— closing open positions."
            )
            return True

    return False

def get_next_news_event(symbol: str) -> dict | None:
    """
    Returns the nearest upcoming high-impact event for the symbol's currencies.
    """
    try:
        df = fetch_forex_factory_calendar()
    except Exception:
        return None
        
    if df is None or df.empty:
        return None
        
    c1 = symbol[:3]
    c2 = symbol[3:6]
    now = datetime.now(timezone.utc)
    
    df_sym = df[df['currency'].isin([c1, c2])].copy()
    if df_sym.empty:
        return None
        
    # Filter upcoming only
    # Ensure tz-aware comparison
    # We must operate safely on series
    future_events = []
    for _, row in df_sym.iterrows():
        dt_news = row['datetime_utc']
        if dt_news.tzinfo is None:
            dt_news = dt_news.replace(tzinfo=timezone.utc)
        if dt_news > now:
            mins = (dt_news - now).total_seconds() / 60.0
            future_events.append({
                "event_name": row['event'],
                "currency": row['currency'],
                "minutes_until": int(mins),
                "datetime_utc": dt_news
            })
            
    if not future_events:
        return None
        
    # Minimum minutes until
    future_events.sort(key=lambda x: x["minutes_until"])
    return future_events[0]

def is_rollover_window() -> bool:
    """
    Returns True if UTC hour == config.ROLLOVER_START_UTC.
    Used to block trading during rollover periods due to high spreads.
    """
    current_utc_hour = datetime.now(timezone.utc).hour
    return current_utc_hour == config.ROLLOVER_START_UTC
