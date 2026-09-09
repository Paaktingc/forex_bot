"""
build_news_events.py

Builds data/news_events.csv (columns: datetime_utc, currency, event) for the
backtest news blackouts. Covers the four MAJOR event families the live filter
widens to ±60 min (NFP, US CPI, FOMC, ECB) from 2015 onward.

Sources (fetched once, parsed from a cache dir):
  - US CPI:  BLS archived news releases; the archive filename encodes the
             RELEASE date (cpi_MMDDYYYY.htm). Fetched via the Wayback Machine
             because bls.gov blocks direct requests. Release 08:30 ET.
  - NFP:     exact empsit_MMDDYYYY dates where archived (2015–2017), plus the
             first-Friday-of-month rule afterwards (validated against the
             exact set; see --validate output). Release 08:30 ET.
  - FOMC:    federalreserve.gov meeting calendars (current + historical
             pages). Statement 14:00 ET on the final meeting day; includes
             unscheduled meetings.
  - ECB:     ecb.europa.eu Governing Council "Monetary policy decisions"
             yearly indexes (dt[isoDate]). Decision 13:45 CET (14:15 CET from
             2022-07-21); press conference 45 min later. Both rows emitted.

Usage:
    python build_news_events.py --cache-dir <dir-with-fetched-html> [--validate]
"""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

NY = ZoneInfo("America/New_York")
BERLIN = ZoneInfo("Europe/Berlin")
UTC = ZoneInfo("UTC")

START_YEAR = 2015

MONTHS = {
    name: i + 1
    for i, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
    )
}
# ECB moved decision publication from 13:45 to 14:15 CET at this meeting
ECB_TIME_CHANGE = date(2022, 7, 21)


def _mmddyyyy_dates(html: str, prefix: str) -> list[date]:
    out = set()
    for token in re.findall(rf"{prefix}_(\d{{8}})", html):
        try:
            parsed = datetime.strptime(token, "%m%d%Y").date()
        except ValueError:
            continue
        if parsed.year >= START_YEAR:
            out.add(parsed)
    return sorted(out)


def parse_cpi(cache: Path) -> list[date]:
    html = (cache / "cpi_full.html").read_text(errors="ignore")
    return _mmddyyyy_dates(html, "cpi")


def parse_nfp_exact(cache: Path) -> list[date]:
    dates: set[date] = set()
    for name in ("empsit_2017id_.html",):
        path = cache / name
        if path.exists():
            dates.update(_mmddyyyy_dates(path.read_text(errors="ignore"), "empsit"))
    return sorted(dates)


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def nfp_rule(year: int, month: int) -> date:
    """
    First Friday of the month, except when that Friday is the 1st — then the
    release slips to the second Friday (BLS needs the full reference week).
    Best variant measured against 36 archived months: 31/36 exact.
    """
    ff = first_friday(year, month)
    return ff + timedelta(days=7) if ff.day <= 1 else ff


def nfp_dates(exact: list[date], through: date) -> tuple[list[date], int, int]:
    """
    Exact archived dates where available; first-Friday rule for later months.
    Returns (dates, rule_hits, rule_total) measuring rule accuracy on the
    exact set.
    """
    exact_by_month = {(d.year, d.month): d for d in exact}
    hits = total = 0
    for (y, m), d in exact_by_month.items():
        total += 1
        if nfp_rule(y, m) == d:
            hits += 1

    out: list[date] = []
    cursor = date(START_YEAR, 1, 1)
    while cursor <= through:
        key = (cursor.year, cursor.month)
        out.append(exact_by_month.get(key, nfp_rule(*key)))
        cursor = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
    return sorted(set(out)), hits, total


def _fomc_statement_day(month_text: str, date_text: str, year: int) -> date | None:
    """
    Meeting labels look like: ('January', '27-28'), ('Apr/May', '30-1'),
    ('March', '3 (unscheduled)'), ('June', '16-17*').
    The statement lands on the LAST meeting day.
    """
    months = [MONTHS[m] for m in re.findall(
        r"January|February|March|April|May|June|July|August|September|October|November|December",
        month_text)]
    if not months:
        # historical h5 style keeps month inside date_text
        months = [MONTHS[m] for m in re.findall(
            r"January|February|March|April|May|June|July|August|September|October|November|December",
            date_text)]
    short = {m[:3]: i for m, i in MONTHS.items()}
    if not months:
        months = [short[m] for m in re.findall(
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", month_text)]
    days = [int(x) for x in re.findall(r"\d{1,2}", date_text)]
    if not months or not days:
        return None
    last_day = days[-1]
    last_month = months[-1]
    # 'Apr/May 30-1': ranges crossing a month boundary use the later month
    if len(months) > 1 and len(days) > 1 and days[-1] < days[0]:
        last_month = months[-1]
    try:
        return date(year, last_month, last_day)
    except ValueError:
        return None


def parse_fomc(cache: Path) -> list[date]:
    dates: set[date] = set()

    current = cache / "fomc_current.html"
    if current.exists():
        soup = BeautifulSoup(current.read_text(errors="ignore"), "html.parser")
        for panel in soup.find_all("div", class_="panel"):
            heading = panel.find(["h4", "h3"])
            if not heading:
                continue
            year_match = re.search(r"(20\d\d) FOMC", heading.get_text(strip=True))
            if not year_match:
                continue
            year = int(year_match.group(1))
            for meeting in panel.find_all("div", class_="fomc-meeting"):
                month_div = meeting.find("div", class_=re.compile("month"))
                date_div = meeting.find("div", class_=re.compile("date"))
                if not month_div or not date_div:
                    continue
                parsed = _fomc_statement_day(
                    month_div.get_text(" ", strip=True),
                    date_div.get_text(" ", strip=True),
                    year,
                )
                if parsed and parsed.year >= START_YEAR:
                    dates.add(parsed)

    for path in sorted(cache.glob("fomc_20??.html")):
        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for tag in soup.find_all("h5"):
            text = tag.get_text(" ", strip=True)
            match = re.match(r"(.+?)\s+Meeting\s*-\s*(20\d\d)", text)
            if not match:
                continue
            body, year = match.group(1), int(match.group(2))
            parsed = _fomc_statement_day(body, body, year)
            if parsed and parsed.year >= START_YEAR:
                dates.add(parsed)

    return sorted(dates)


def parse_ecb(cache: Path) -> list[date]:
    dates: set[date] = set()
    for path in sorted(cache.glob("ecb_20??.html")):
        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for dt_tag in soup.find_all("dt"):
            iso = dt_tag.get("isodate") or dt_tag.get("isoDate")
            dd = dt_tag.find_next_sibling("dd")
            if not iso or dd is None:
                continue
            if "monetary policy decisions" not in dd.get_text(" ", strip=True).lower():
                continue
            parsed = date.fromisoformat(iso)
            if parsed.year >= START_YEAR:
                dates.add(parsed)
    return sorted(dates)


def _row(day: date, hour: int, minute: int, tz: ZoneInfo, currency: str, event: str) -> dict:
    stamp = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)
    return {
        "datetime_utc": stamp.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "currency": currency,
        "event": event,
    }


def build(cache: Path, out_path: Path, validate: bool = False) -> pd.DataFrame:
    cpi = parse_cpi(cache)
    nfp_exact = parse_nfp_exact(cache)
    fomc = parse_fomc(cache)
    ecb = parse_ecb(cache)

    through = max(cpi[-1], fomc[-1], ecb[-1])
    nfp, rule_hits, rule_total = nfp_dates(nfp_exact, through)

    rows: list[dict] = []
    for day in nfp:
        rows.append(_row(day, 8, 30, NY, "USD", "Non-Farm Employment Change"))
    for day in cpi:
        rows.append(_row(day, 8, 30, NY, "USD", "CPI m/m"))
    for day in fomc:
        rows.append(_row(day, 14, 0, NY, "USD", "FOMC Statement"))
    for day in ecb:
        decision_time = (13, 45) if day < ECB_TIME_CHANGE else (14, 15)
        rows.append(_row(day, *decision_time, BERLIN, "EUR", "ECB Main Refinancing Rate"))
        presser = (decision_time[0], decision_time[1] + 45)
        hour, minute = presser[0] + presser[1] // 60, presser[1] % 60
        rows.append(_row(day, hour, minute, BERLIN, "EUR", "ECB Press Conference"))

    df = pd.DataFrame(rows).sort_values("datetime_utc").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"NFP:  {len(nfp)} dates ({len(nfp_exact)} exact; first-Friday rule "
          f"matched {rule_hits}/{rule_total} archived months)")
    print(f"CPI:  {len(cpi)} dates (exact, BLS archive filenames)")
    print(f"FOMC: {len(fomc)} dates (Fed calendars incl. unscheduled)")
    print(f"ECB:  {len(ecb)} decision dates (2 rows each: decision + presser)")
    print(f"Total rows: {len(df)} → {out_path}")

    if validate:
        per_year = pd.to_datetime(df["datetime_utc"]).dt.year.value_counts().sort_index()
        print("\nRows per year:")
        print(per_year.to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True,
                        help="Directory with fetched source HTML files.")
    parser.add_argument("--out", default="data/news_events.csv")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    build(Path(args.cache_dir), Path(args.out), validate=args.validate)


if __name__ == "__main__":
    main()
