# XAUUSD copier teardown — reverse-engineering study

Self-contained analysis of a third-party XAUUSD-STDc copy-trading account
(29498319, USC cent account, 29 Jun – 23 Jul 2026), performed to decide whether
any part of it is usable on a The5ers Bootcamp challenge.

**Verdict: NO-GO on the original bot. See `XAUUSD_Bot_Reverse_Engineering_Report.md` §13.**

This folder is **research only**. Nothing here is imported by the live bot and
`research_backtest.py` refuses to run if any broker SDK is loaded in-process.

## Layout

```
XAUUSD_Bot_Reverse_Engineering_Report.md   full 13-deliverable report
research_backtest.py                       tick-level backtester for the
                                           independent risk-capped redesign (§10)
charts/charts_overview.png                 12 diagnostic charts
data/                                      reconstructed + raw datasets
pipeline/                                  the scripts that produced data/
logs/                                      full numeric audit trail
```

## Reproducing

```bash
cd pipeline
python3 parse_statement.py            # xlsx -> raw_*.csv
python3 reconstruct.py                # -> trades_reconstructed.csv, baskets_*.csv
python3 analyse_sizing_grid_exit.py   # sizing / grid / exit / copier / timing models
python3 tick_analysis.py              # timezone calibration, MAE/MFE, floating DD
python3 entry_signal_and_charts.py    # entry-signal precision-recall + charts
```

Paths at the top of each script point at the original inputs
(`xauusd trading lot.xlsx`, Dukascopy tick CSV) and must be repointed.

## Headline findings

| | |
|---|---|
| Round-trip positions reconstructed | 1,356 — **100% matched to 0.00000 USD** price error |
| Daily P/L reconciliation | **exact on 15 of 16 supplied sheets** |
| Copy-trading | **2,721 / 2,721 deals** carry `copy #<master ticket>` |
| Stop-losses | **0 / 2,721** |
| Lot ladder | `0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.11,0.14,0.18,0.23,0.30,0.39,0.51,0.67,0.87` (88.1% of baskets exact) |
| Grid step | median **2.10 USD/oz** adverse, non-expanding |
| Basket exit | ≈ **+2.3 USD/oz** from weighted-average entry |
| Server clock | **UTC+3** (calibrated against Dukascopy, 0.235 USD residual) |
| Contract | **1 oz/lot**, P/L in US cents, leverage **1:500** |
| Closed-balance drawdown | **0.00%** |
| **Tick-reconstructed floating drawdown** | **−7.67% of balance** |

The 0.00% / −7.67% pair is the whole story: a 90.3% win rate produced by never
realising a loss. This is the same failure mode as the in-sample artifacts
documented in `research_log.md` — a number that looks like edge but is really
a deferred liability.

## Relevance to this repo

1. **Instrument diversification (Lever 1)** — the XAUUSD tick data and the
   `symbol_specs`-style facts derived here (contract size, spread distribution,
   session profile) feed the multi-instrument pooling work. Median spread
   0.68 USD, p99 0.97, max 15.00; 84% of ticks fall in the 07:00–20:00 UTC+3
   window where the copier also concentrated 94.1% of its entries.
2. **Risk-layer reuse** — the redesign in §10 is deliberately specified against
   the existing `risk_manager.py` primitives (step-start anchor, daily/total
   kill switch, persisted state) rather than inventing a parallel risk stack.
3. **Do not adopt the ladder.** The geometric 1.30 progression is the single
   most dangerous component and is explicitly excluded from the redesign, which
   uses a bounded arithmetic ladder sized backwards from a fixed stop.
