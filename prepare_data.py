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
PRICE_MIN = 1.00
PRICE_MAX = 1.25
MIN_M15_BARS = 35_000


def find_raw_files() -> list[Path]:
    files = sorted(
        file for file in RAW_DIR.rglob("*.csv") if "EURUSD" in file.name
    )
    if not files:
        raise FileNotFoundError(f"No raw EURUSD CSV files found in {RAW_DIR}")
    return files


def load_histdata_csv(file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    try:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str).str.zfill(6),
            format="%Y%m%d %H%M%S",
        )
    except ValueError:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M",
        )
    df.set_index("datetime", inplace=True)
    df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def load_all_m1_data() -> pd.DataFrame:
    frames = [load_histdata_csv(file) for file in find_raw_files()]
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


def validate_frame(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> None:
    if len(df_m15) < MIN_M15_BARS:
        raise ValueError(
            f"M15 validation failed: expected at least {MIN_M15_BARS} bars, found {len(df_m15)}"
        )

    for name, df in {"M15": df_m15, "H1": df_h1}.items():
        if df.isna().any().any():
            raise ValueError(f"{name} validation failed: NaN rows detected")

        price_cols = ["open", "high", "low", "close"]
        min_price = float(df[price_cols].min().min())
        max_price = float(df[price_cols].max().max())
        if min_price < PRICE_MIN or max_price > PRICE_MAX:
            raise ValueError(
                f"{name} validation failed: prices out of EURUSD range ({min_price:.5f} - {max_price:.5f})"
            )


def save_outputs(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> None:
    df_m15.to_csv(M15_OUTPUT)
    df_h1.to_csv(H1_OUTPUT)


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_m1 = load_all_m1_data()
    df_m15 = resample_ohlcv(df_m1, "15min")
    df_h1 = resample_ohlcv(df_m1, "1h")
    validate_frame(df_m15, df_h1)
    save_outputs(df_m15, df_h1)

    print(f"M15: {len(df_m15)} bars | H1: {len(df_h1)} bars | {df_m15.index.min()} → {df_m15.index.max()}")
    print(f"Raw CSV files loaded: {len(find_raw_files())}")
    print(f"Raw M1 rows merged: {len(df_m1)}")
    print(f"Saved: {M15_OUTPUT}")
    print(f"Saved: {H1_OUTPUT}")
    return df_m15, df_h1


if __name__ == "__main__":
    run_pipeline()
