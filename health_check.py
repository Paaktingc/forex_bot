"""
health_check.py

Basic live-environment checks for a Windows VPS deployment.
"""

from __future__ import annotations

import logging
from typing import Callable

import config
import data_feed
import news_filter
from model import load_model
from risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _run_check(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def _check_mt5_connection() -> None:
    balance = data_feed.connect_mt5()
    if balance <= 0:
        raise RuntimeError(f"Unexpected MT5 balance returned: {balance}")


def _check_data_feed() -> None:
    df = data_feed.get_ohlcv(config.SYMBOL, "M15", 10)
    if df.empty or len(df) < 10:
        raise RuntimeError(f"Expected 10 bars, received {len(df)}")


def _check_model_load() -> None:
    model, encoder = load_model()
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not expose predict_proba")
    if len(getattr(encoder, "classes_", [])) == 0:
        raise RuntimeError("Label encoder has no classes")


def _check_news_filter() -> None:
    news_filter.fetch_forex_factory_calendar()


def _check_risk_manager() -> None:
    account = data_feed.get_account_info()
    balance = float(account["balance"])
    rm = RiskManager(balance)
    can_trade, _ = rm.can_trade(float(account["equity"]), 0)
    if not isinstance(can_trade, bool):
        raise RuntimeError("Risk manager did not return a boolean trade decision")


def main() -> None:
    checks = [
        ("MT5 connection", _check_mt5_connection),
        ("Data feed", _check_data_feed),
        ("Model load", _check_model_load),
        ("News filter", _check_news_filter),
        ("Risk manager", _check_risk_manager),
    ]

    results = [_run_check(name, fn) for name, fn in checks]

    if all(results):
        print("VPS READY FOR LIVE TRADING")
    else:
        print("VPS NOT READY")


if __name__ == "__main__":
    main()
