"""
Symbol specifications used for pip math and conservative risk sizing.
"""

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    pip_size: float
    pip_value_per_standard_lot: float
    min_lot: float
    max_lot: float
    lot_step: float


_SPECS: dict[str, SymbolSpec] = {
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        pip_size=0.0001,
        pip_value_per_standard_lot=10.0,
        min_lot=0.01,
        max_lot=5.0,
        lot_step=0.01,
    ),
}


def normalize_symbol(symbol: str | None) -> str:
    return (symbol or config.SYMBOL).upper()


def get_pip_size(symbol: str | None = None) -> float:
    symbol = normalize_symbol(symbol)
    if symbol in _SPECS:
        return _SPECS[symbol].pip_size
    return 0.01 if symbol.endswith("JPY") else 0.0001


def get_symbol_spec(symbol: str | None = None) -> SymbolSpec | None:
    """Return a lot-sizing spec for symbols explicitly supported for trading."""
    return _SPECS.get(normalize_symbol(symbol))
