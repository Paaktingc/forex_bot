"""Download and standardise M1 forex OHLCV data for backtesting.

Primary source: HistData.com MT/ASCII archives.
Fallback source: MetaTrader 5 for current-year/YTD coverage.
"""

from __future__ import annotations

import argparse
import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - fallback remains optional until used.
    mt5 = None


BASE_URL = "https://www.histdata.com"
OUTPUT_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
TODAY_UTC = datetime.now(UTC).date()

REQUEST_DELAY_SECONDS = 2
HTTP_TIMEOUT_SECONDS = 60
FOREX_START_WEEKDAY = 0
FOREX_END_WEEKDAY = 4
GAP_THRESHOLD = pd.Timedelta(minutes=5)

TARGETS = {
    "EURUSD": [2024],
    "GBPUSD": [2023, 2024, 2025, 2026],
    "USDJPY": [2023, 2024, 2025, 2026],
    "AUDUSD": [2023, 2024, 2025, 2026],
    "XAUUSD": [2023, 2024, 2025, 2026],
}


@dataclass(frozen=True)
class DownloadUnit:
    pair: str
    year: int
    month: int | None = None

    @property
    def label(self) -> str:
        if self.month is None:
            return f"{self.pair} {self.year}"
        return f"{self.pair} {self.year}-{self.month:02d}"

    @property
    def period_value(self) -> str:
        if self.month is None:
            return str(self.year)
        return f"{self.year}{self.month:02d}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    summary_rows: list[dict[str, object]] = []
    all_gaps: list[pd.DataFrame] = []

    for pair, years in TARGETS.items():
        for year in years:
            print(f"\n=== {pair} {year} ===")
            df = collect_year(pair, year, session, raw_dir, args.skip_histdata)
            if year == TODAY_UTC.year:
                df = merge_mt5_ytd(pair, year, df, args.skip_mt5)

            if df.empty:
                print(f"No bars collected for {pair} {year}; writing no output file.")
                summary_rows.append(
                    {"pair": pair, "year": year, "total_bars": 0, "gap_count": 0}
                )
                continue

            df = standardise_frame(df, year)
            out_file = output_dir / f"{pair}_M1_{year}.csv"
            df.to_csv(out_file, index=False)

            gaps = find_gaps(df, pair, year)
            if not gaps.empty:
                all_gaps.append(gaps)

            summary_rows.append(
                {
                    "pair": pair,
                    "year": year,
                    "total_bars": len(df),
                    "gap_count": len(gaps),
                }
            )
            print(
                f"Saved {out_file} | bars={len(df):,} | gaps>{GAP_THRESHOLD}={len(gaps):,}"
            )

    gaps_report = (
        pd.concat(all_gaps, ignore_index=True)
        if all_gaps
        else pd.DataFrame(
            columns=[
                "pair",
                "year",
                "gap_start",
                "gap_end",
                "gap_minutes",
                "missing_minutes",
            ]
        )
    )
    gaps_report.to_csv(output_dir / "gaps_report.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    print("\nSummary")
    if summary.empty:
        print("No targets processed.")
    else:
        print(summary.to_string(index=False))
    print(f"\nGap report: {output_dir / 'gaps_report.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HistData M1 forex data and standardise it for backtests."
    )
    parser.add_argument("--output-dir", default="m1_data")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--skip-histdata",
        action="store_true",
        help="Do not request HistData; useful for testing MT5 fallback only.",
    )
    parser.add_argument(
        "--skip-mt5",
        action="store_true",
        help="Do not use MT5 fallback for current-year/YTD data.",
    )
    return parser.parse_args()


def collect_year(
    pair: str, year: int, session: requests.Session, raw_dir: Path, skip_histdata: bool
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    local_frames = read_existing_raw_files(raw_dir, pair, year)
    if local_frames:
        frames.extend(local_frames)
        print(f"Loaded {len(local_frames)} local raw file(s) from {raw_dir}.")

    if skip_histdata:
        return combine_frames(frames)

    if year == TODAY_UTC.year:
        units = [
            DownloadUnit(pair, year, month)
            for month in range(1, TODAY_UTC.month + 1)
        ]
    else:
        units = [DownloadUnit(pair, year)]

    for unit in units:
        try:
            frame = download_histdata_unit(session, unit, raw_dir)
        except Exception as exc:
            print(f"HistData failed for {unit.label}: {exc}")
            continue

        if not frame.empty:
            frames.append(frame)
            print(f"HistData loaded {unit.label}: {len(frame):,} bars.")
        time.sleep(REQUEST_DELAY_SECONDS)

    return combine_frames(frames)


def read_existing_raw_files(raw_dir: Path, pair: str, year: int) -> list[pd.DataFrame]:
    if not raw_dir.exists():
        return []

    candidates = sorted(
        path
        for path in raw_dir.rglob(f"*{pair}*M1*{year}*.csv")
        if path.is_file()
    )
    return [parse_histdata_csv(path) for path in candidates]


def download_histdata_unit(
    session: requests.Session, unit: DownloadUnit, raw_dir: Path
) -> pd.DataFrame:
    page_url = histdata_page_url(unit)
    response = session.get(page_url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()

    post_url, payload = build_histdata_post(response.text, response.url, unit)
    download = session.post(
        post_url,
        data=payload,
        headers={"Referer": response.url},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    download.raise_for_status()

    content_type = download.headers.get("Content-Type", "")
    if not download.content.startswith(b"PK"):
        raise RuntimeError(
            f"download did not return a zip archive "
            f"(content-type={content_type!r}, size={len(download.content)})"
        )

    extract_dir = raw_dir / histdata_archive_stem(unit)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        archive.extractall(extract_dir)

    frames = [
        parse_histdata_csv(path)
        for path in sorted(extract_dir.glob("*.csv"))
        if path.is_file()
    ]
    return combine_frames(frames)


def histdata_page_url(unit: DownloadUnit) -> str:
    parts = [
        BASE_URL,
        "download-free-forex-historical-data",
        "?",
        "ascii",
        "1-minute-bar-quotes",
        unit.pair,
        str(unit.year),
    ]
    if unit.month is not None:
        parts.append(f"{unit.month:02d}")
    return "/".join(parts)


def histdata_archive_stem(unit: DownloadUnit) -> str:
    suffix = unit.period_value
    return f"HISTDATA_COM_MT_{unit.pair}_M1{suffix}"


def build_histdata_post(
    html: str, page_url: str, unit: DownloadUnit
) -> tuple[str, dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    form = choose_download_form(soup)
    if form is None:
        raise RuntimeError("could not find HistData download form")

    action = form.get("action") or "/get.php"
    post_url = urljoin(page_url, action)
    payload: dict[str, str] = {}

    for input_tag in form.find_all("input"):
        name = input_tag.get("name")
        if not name:
            continue
        payload[name] = input_tag.get("value", "")

    token = find_csrf_token(soup, payload)
    if token:
        payload.setdefault("tk", token)

    payload["date"] = str(unit.year)
    payload["datemonth"] = unit.period_value
    payload.setdefault("platform", "ASCII")
    payload["timeframe"] = "M1"
    payload["fxpair"] = unit.pair
    return post_url, payload


def choose_download_form(soup: BeautifulSoup):
    forms = soup.find_all("form")
    if not forms:
        return None
    for form in forms:
        action = (form.get("action") or "").lower()
        text = form.get_text(" ", strip=True).lower()
        if "get.php" in action or "download" in text:
            return form
    return forms[0]


def find_csrf_token(soup: BeautifulSoup, payload: dict[str, str]) -> str | None:
    for key in ("tk", "token", "csrf_token", "_csrf", "_token"):
        if payload.get(key):
            return payload[key]

    text = soup.get_text("\n")
    match = re.search(r"(?:tk|token|csrf[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1)
    return None


def parse_histdata_csv(path: Path) -> pd.DataFrame:
    sep = sniff_separator(path)
    frame = pd.read_csv(path, sep=sep, header=None, dtype=str)

    if frame.shape[1] < 6:
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected at least 6")

    if frame.shape[1] >= 7:
        frame = frame.iloc[:, :7].copy()
        frame.columns = ["date", "time", "open", "high", "low", "close", "volume"]
        frame["datetime"] = pd.to_datetime(
            frame["date"].str.strip() + " " + frame["time"].str.strip(),
            format="%Y.%m.%d %H:%M",
            utc=True,
            errors="coerce",
        )
    else:
        frame = frame.iloc[:, :6].copy()
        frame.columns = ["datetime", "open", "high", "low", "close", "volume"]
        frame["datetime"] = pd.to_datetime(
            frame["datetime"].str.strip(),
            utc=True,
            errors="coerce",
        )
    return clean_ohlcv(frame)


def sniff_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.readline()
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        raise ValueError(f"could not detect delimiter for {path}")
    return sep


def merge_mt5_ytd(pair: str, year: int, frame: pd.DataFrame, skip_mt5: bool) -> pd.DataFrame:
    if skip_mt5:
        return frame

    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime.combine(TODAY_UTC, datetime.max.time(), tzinfo=UTC)

    if not frame.empty:
        latest = pd.to_datetime(frame["datetime"], utc=True).max()
        if latest.date() >= TODAY_UTC:
            return frame
        start = latest.to_pydatetime() + pd.Timedelta(minutes=1)

    try:
        mt5_frame = load_mt5_rates(pair, start, end)
    except Exception as exc:
        print(f"MT5 fallback failed for {pair} {year}: {exc}")
        return frame

    if mt5_frame.empty:
        print(f"MT5 fallback returned no bars for {pair} {year}.")
        return frame

    print(
        f"MT5 fallback loaded {pair} {year}: {len(mt5_frame):,} bars "
        f"from {start.isoformat()} to {end.isoformat()}."
    )
    return combine_frames([frame, mt5_frame])


def load_mt5_rates(pair: str, start: datetime, end: datetime) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not importable")

    initialized_here = False
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    initialized_here = True

    try:
        symbol = resolve_mt5_symbol(pair)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    finally:
        if initialized_here:
            mt5.shutdown()

    if rates is None:
        raise RuntimeError(f"mt5.copy_rates_range failed: {mt5.last_error()}")
    if len(rates) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = pd.DataFrame(rates)
    frame["datetime"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["volume"] = frame.get("tick_volume", 0)
    return clean_ohlcv(frame)


def resolve_mt5_symbol(pair: str) -> str:
    if mt5.symbol_select(pair, True):
        return pair

    symbols = mt5.symbols_get()
    if symbols is None:
        raise RuntimeError(f"could not list MT5 symbols: {mt5.last_error()}")

    pair_upper = pair.upper()
    for symbol in symbols:
        name = symbol.name
        if name.upper() == pair_upper or name.upper().startswith(pair_upper):
            if mt5.symbol_select(name, True):
                return name
    raise RuntimeError(f"could not select MT5 symbol for {pair}")


def clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    clean["datetime"] = pd.to_datetime(clean["datetime"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["datetime", "open", "high", "low", "close"])
    clean = clean[OUTPUT_COLUMNS]
    clean = clean.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return clean.reset_index(drop=True)


def standardise_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    clean = clean_ohlcv(frame)
    start = pd.Timestamp(datetime(year, 1, 1, tzinfo=UTC))
    if year == TODAY_UTC.year:
        end = pd.Timestamp(datetime.combine(TODAY_UTC, datetime.max.time(), tzinfo=UTC))
    else:
        end = pd.Timestamp(datetime(year, 12, 31, 23, 59, tzinfo=UTC))

    clean = clean[(clean["datetime"] >= start) & (clean["datetime"] <= end)].copy()
    clean["datetime"] = clean["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return clean[OUTPUT_COLUMNS].reset_index(drop=True)


def combine_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return clean_ohlcv(pd.concat(non_empty, ignore_index=True))


def find_gaps(frame: pd.DataFrame, pair: str, year: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(frame["datetime"], utc=True).sort_values()
    trading_timestamps = timestamps[timestamps.map(is_forex_trading_minute)]
    diffs = trading_timestamps.diff()
    candidate_indexes = trading_timestamps.index[diffs > GAP_THRESHOLD]
    gap_rows: list[dict[str, object]] = []

    for index in candidate_indexes:
        previous_ts = trading_timestamps.shift(1).loc[index]
        current_ts = trading_timestamps.loc[index]
        missing_minutes = count_missing_trading_minutes(previous_ts, current_ts)
        if missing_minutes <= 5:
            continue

        gap_rows.append(
            {
                "pair": pair,
                "year": year,
                "gap_start": previous_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gap_end": current_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gap_minutes": missing_minutes + 1,
                "missing_minutes": missing_minutes,
            }
        )

    if not gap_rows:
        return pd.DataFrame(
            columns=[
                "pair",
                "year",
                "gap_start",
                "gap_end",
                "gap_minutes",
                "missing_minutes",
            ]
        )

    return pd.DataFrame(gap_rows)


def count_missing_trading_minutes(previous_ts: pd.Timestamp, current_ts: pd.Timestamp) -> int:
    start = previous_ts + pd.Timedelta(minutes=1)
    end = current_ts - pd.Timedelta(minutes=1)
    if start > end:
        return 0

    total = 0
    day_start = start.normalize()
    day_end = end.normalize()
    day = day_start

    while day <= day_end:
        if FOREX_START_WEEKDAY <= day.weekday() <= FOREX_END_WEEKDAY:
            window_start = max(start, day)
            window_end = min(end, day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1))
            if window_start <= window_end:
                total += int((window_end - window_start) / pd.Timedelta(minutes=1)) + 1
        day += pd.Timedelta(days=1)

    return total


def is_forex_trading_minute(ts: pd.Timestamp) -> bool:
    weekday = ts.weekday()
    return FOREX_START_WEEKDAY <= weekday <= FOREX_END_WEEKDAY


if __name__ == "__main__":
    main()
