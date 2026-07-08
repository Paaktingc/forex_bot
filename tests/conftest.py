import sys
from unittest.mock import MagicMock

class MockMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_MONEY = 10019
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60
    
    @staticmethod
    def initialize(*args, **kwargs): return True
    @staticmethod
    def login(*args, **kwargs): return True
    @staticmethod
    def shutdown(): pass
    @staticmethod
    def last_error(): return (1, "Mocked error")

# Mock Windows-only MetaTrader5 library so tests can be collected on Mac
sys.modules['MetaTrader5'] = MockMT5()

# Real ML libraries will be used for testing.

import pytest


@pytest.fixture(autouse=True)
def _isolate_risk_flag_files(tmp_path, monkeypatch):
    """Keep RiskManager persistence files out of the real logs/ directory."""
    import config

    monkeypatch.setattr(
        config, "RISK_DISABLED_FLAG_PATH", str(tmp_path / "trading_disabled.json")
    )
    monkeypatch.setattr(config, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))
