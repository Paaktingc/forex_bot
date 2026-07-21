"""
prepare_data.py

Builds consolidated EURUSD M15 and H1 datasets from HistData M1 CSV files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
M15_OUTPUT = PROJECT_ROOT / "data" / "EURUSD_M15_real.csv"
H1_OUTPUT = PROJECT_ROOT / "data" / "EURUSD_H1_real.csv"
MIN_M15_BARS = 35_000

# Per-symbol sanity bounds for the 2015–present window
SYMBOL_BOUNDS: dict[str, tuple[float, float]] = {
    "EURUSD": (0.90, 1.65),      # incl. 2008 high ≈ 1.60 (2007+ archive)
    "GBPUSD": (1.00, 2.20),      # incl. 2007 high ≈ 2.11
    "AUDUSD": (0.54, 0.90),      # 2020 low ≈ 0.551, 2018 high ≈ 0.81
    "USDJPY": (90.0, 170.0),     # 2016 low ≈ 98.9, 2024 high ≈ 161.9
    "XAUUSD": (900.0, 6500.0),   # 2015 low ≈ 1046; 2026 high ≈ 5597
    "GRXEUR": (7500.0, 27000.0), # DAX proxy; COVID low ≈ 7969
    "ETXEUR": (2200.0, 6500.0),  # EURO STOXX 50 proxy, 2020 low ≈ 2300
}

# HistData feed-contamination guards (research_log.md cycle 5): the GRXEUR
# archive quotes EURO STOXX-scale prices between 2020-06-15 and 2023-12-03
# (provider switch). Only the verified-DAX-scale windows are kept.
SYMBOL_CLEAN_WINDOWS: dict[str, list[tuple[str, str]]] = {
    "GRXEUR": [("2015-01-01", "2020-06-14"), ("2023-12-04", "2030-01-01")],
}
PRICE_MIN, PRICE_MAX = SYMBOL_BOUNDS["EURUSD"]  # legacy aliases


def find_raw_files(symbol: str = "EURUSD") -> list[Path]:
    files = sorted(
        file for file in RAW_DIR.rglob("*.csv") if symbol in file.name
    )
    if not files:
        raise FileNotFoundError(f"No raw {symbol} CSV files found in {RAW_DIR}")
    return files


HISTDATA_TZ = "America/New_York"


def load_histdata_csv(file: Path) -> pd.DataFrame:
    """
    Loads a HistData M1 file (either the semicolon 'ASCII' layout with a
    single 'YYYYMMDD HHMMSS' field, or the comma 'MT' layout with separate
    date/time fields).

    IMPORTANT timezone note: HistData timestamps are US EASTERN local time —
    the market opens Sunday 17:00 and closes Friday 16:59 file-time in both
    winter and summer (verified across 2015/2023/2024 files). They were
    previously mislabelled as UTC, shifting every session by 4–5 hours.
    """
    with open(file, "r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()

    if ";" in first_line:  # ASCII layout: '20150101 130000;o;h;l;c;v'
        df = pd.read_csv(
            file,
            sep=";",
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(
            df["datetime"].astype(str), format="%Y%m%d %H%M%S"
        )
    else:  # MT layout: '2023.01.01,17:04,o,h,l,c,v'
        df = pd.read_csv(
            file,
            header=None,
            names=["date", "time", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M",
        )

    df.set_index("datetime", inplace=True)
    # DST transitions happen 02:00 Sunday local when the market is closed,
    # so ambiguous/nonexistent stamps should not occur; drop any that do.
    df.index = df.index.tz_localize(
        HISTDATA_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    df = df[df.index.notna()]
    df.index = df.index.tz_convert("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def load_all_m1_data(symbol: str = "EURUSD") -> pd.DataFrame:
    frames = [load_histdata_csv(file) for file in find_raw_files(symbol)]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return (
        df.resample(timeframe)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def validate_frame(
    df_m15: pd.DataFrame, df_h1: pd.DataFrame, symbol: str = "EURUSD"
) -> None:
    if len(df_m15) < MIN_M15_BARS:
        raise ValueError(
            f"M15 validation failed: expected at least {MIN_M15_BARS} bars, found {len(df_m15)}"
        )

    price_min, price_max = SYMBOL_BOUNDS[symbol]
    for name, df in {"M15": df_m15, "H1": df_h1}.items():
        if df.isna().any().any():
            raise ValueError(f"{name} validation failed: NaN rows detected")

        price_cols = ["open", "high", "low", "close"]
        min_price = float(df[price_cols].min().min())
        max_price = float(df[price_cols].max().max())
        if min_price < price_min or max_price > price_max:
            raise ValueError(
                f"{name} validation failed: prices out of {symbol} range "
                f"({min_price:.5f} - {max_price:.5f})"
            )


def run_pipeline(symbol: str = "EURUSD") -> tuple[pd.DataFrame, pd.DataFrame]:
    df_m1 = load_all_m1_data(symbol)
    windows = SYMBOL_CLEAN_WINDOWS.get(symbol)
    if windows:
        keep = pd.Series(False, index=df_m1.index)
        for start, end in windows:
            keep |= (df_m1.index >= pd.Timestamp(start, tz="UTC")) & (
                df_m1.index <= pd.Timestamp(end, tz="UTC")
            )
        dropped = int((~keep).sum())
        df_m1 = df_m1[keep]
        print(f"{symbol}: dropped {dropped} contaminated M1 rows outside clean windows")
    df_m15 = resample_ohlcv(df_m1, "15min")
    df_h1 = resample_ohlcv(df_m1, "1h")
    validate_frame(df_m15, df_h1, symbol)

    m15_path = PROJECT_ROOT / "data" / f"{symbol}_M15_real.csv"
    h1_path = PROJECT_ROOT / "data" / f"{symbol}_H1_real.csv"
    df_m15.to_csv(m15_path)
    df_h1.to_csv(h1_path)

    print(f"{symbol} M15: {len(df_m15)} bars | H1: {len(df_h1)} bars | "
          f"{df_m15.index.min()} → {df_m15.index.max()}")
    print(f"Raw CSV files loaded: {len(find_raw_files(symbol))}")
    print(f"Saved: {m15_path} / {h1_path}")
    return df_m15, df_h1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD", choices=sorted(SYMBOL_BOUNDS))
    parser.add_argument("--all", action="store_true", help="Build every symbol.")
    args = parser.parse_args()
    for sym in (sorted(SYMBOL_BOUNDS) if args.all else [args.symbol]):
        run_pipeline(sym)
