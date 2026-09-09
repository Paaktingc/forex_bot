# Provisional risk-default status (2026-07-29)

`config.PROGRAMMES[*]["risk_per_trade_pct"] = 0.002` (0.20%) in the uncommitted
Cycle-6 working tree is a **conservative provisional default, not validated for
deployment**. Its original justification — a P(maxDD>5%)<5% feasibility envelope
built on the H1-regime return distribution — was **invalidated** when that signal
was shown to be a higher-timeframe look-ahead artifact (see `research_log.md`,
Gate 0 / commit `9b49079` and the alignment-fix / retirement commits).

- 0.20% is **not** described here as an approved or optimal risk level.
- It is **not changed** in this task; the H1 line it was derived for is retired,
  and no live strategy currently depends on it.
- Any future deployment must re-derive the risk-per-trade from a *validated*,
  leak-free return distribution before this value is trusted.

The committed-baseline test suite does not require this value to run
(`tests/test_risk_manager.py` was decoupled from the mutable default in commit
`a796777`).
