"""Broker adapter factory."""

from __future__ import annotations

from typing import Optional

import config
from .base import BrokerAdapter

_broker: Optional[BrokerAdapter] = None
_broker_name: Optional[str] = None


def get_broker() -> BrokerAdapter:
    """Return the configured broker adapter singleton."""
    global _broker, _broker_name
    broker_name = config.BROKER.lower().strip()
    if _broker is not None and _broker_name == broker_name:
        return _broker

    if broker_name == "mt5":
        from .mt5_adapter import MT5BrokerAdapter

        _broker = MT5BrokerAdapter()
    elif broker_name == "ctrader":
        from .ctrader_adapter import CTraderBrokerAdapter

        _broker = CTraderBrokerAdapter()
    else:
        raise ValueError(f"Unsupported BROKER={config.BROKER!r}; use 'mt5' or 'ctrader'.")

    _broker_name = broker_name
    return _broker


def reset_broker() -> None:
    """Reset the cached broker adapter. Intended for tests."""
    global _broker, _broker_name
    if _broker is not None:
        _broker.shutdown()
    _broker = None
    _broker_name = None
