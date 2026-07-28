# FX Intraday Volatility-Expansion Continuation — signal validation

**Status: STOP — no directional edge (Gate 3 failed on discovery).** New,
independent research line; inherits nothing from the retired H1 regime strategy
or the XAUUSD basket.

## Hypothesis

> After a genuine intraday volatility expansion during liquid London / London–NY
> overlap hours, price tends to continue in the expansion direction over the
> next 1–4 hours, after realistic costs.

## Frozen specification (Gate 2 — declared before any result, `config.py`)

- Timeframe: M15 primary (M30 reserved for a Gate-4 robustness comparison).
- `ATR_reference[t]` = ATR(20) of true range **through t−1** (the classified bar
  is excluded from its own reference).
- Expansion: `true_range[t] >= 1.5 × ATR_reference[t]`.
- Direction: bullish `close>open`, bearish `close<open` (zero-body ignored).
- Close location: bull `(close−low)/(high−low) >= 0.75`; bear mirrored; zero-range rejected.
- Breakout: bull `close > max(high[t−8..t−1])`; bear `close < min(low[t−8..t−1])` (current bar excluded).
- Session: expansion bar **close** within 08:00–16:00 Europe/London (DST-aware).
- Cooldown: ≤1 entry per pair per rolling 4h (from signal timestamp).
- Entry: next M15 bar open + spread + slippage; reject if the next bar is missing.
- Stop `1.0 × ATR_reference`; target `1.5R`; time exit after 16 bars; same-bar
  stop-and-target → **stop first**. Verdicts use **after-cost** results.

## Causality (the point of this line)

Every feature uses only closed bars ≤ decision time; the ATR reference excludes
the current bar; breakout/close-location use only prior closed bars; entry is
strictly the next bar's open. Enforced by `tests/research/test_intraday_volatility_continuation.py`
(ATR-reference exclusion, truncation-invariance of the signal, next-bar entry,
DST session, USDJPY pip scaling, lockbox guard, no negative shifts). The locked
OOS period (2023→) is sealed by `prepare_data.load_split(..., unlock_oos=…)`.

## Data (Gate 1 — `results/gate1_audit.txt`, PASS)

EURUSD/GBPUSD 2007→2026, AUDUSD/USDJPY 2015→2026; all M15, UTC, left-labelled,
monotonic, zero dups/nulls/impossible-OHLC. USDJPY pip 0.01 (correct). No spread
column — spreads modelled from repo floors. Splits: discovery 2007–2018,
validation 2019–2022, locked OOS 2023→ (untouched).

## Discovery result (Gate 3 — `results/gate3_discovery.txt`, FAIL)

After-cost, discovery window:

| Pair | n | /yr | E[R] net | E[R] gross | PF | WR | iid 95% CI |
|---|---|---|---|---|---|---|---|
| EURUSD | 4107 | 342 | −0.1819 | −0.0470 | 0.751 | 0.380 | [−0.221, −0.143] |
| GBPUSD | 3959 | 330 | −0.2363 | −0.1057 | 0.682 | 0.350 | [−0.275, −0.197] |
| AUDUSD | 1232 | 308 | −0.3378 | −0.1280 | 0.586 | 0.338 | [−0.408, −0.269] |
| USDJPY | 1257 | 314 | −0.3784 | −0.1645 | 0.547 | 0.329 | [−0.447, −0.310] |

- **Pooled 4-pair −0.2439R** (PF 0.678, CI [−0.269, −0.220]); **non-EURUSD −0.2834R**.
- Every pair and (almost) every calendar year negative; removing the best pair
  or best year does not help; winners are diffuse (top 1% = 1.0% of gross wins),
  so it is not a few-trade artifact. Frequency ~1320 trades/yr (bucket >600).
- **Baselines:** matched-random pct-rank **11.3** (P(random ≥ strategy)=0.887 — the
  signal is *worse* than random-matched entries under the same management);
  shuffled-label P=0.062 (not beaten at 95%); unconditional session continuation
  −0.2581 (the complex signal adds only +0.014R).

**Why it fails.** Random in-session entries under this management are ~fair
(−0.008R gross), so there is no management bug. The expansion bar carries only a
tiny raw tendency (+0.025R gross measured *at* the signal bar), but it is
**intrabar**: a realistic next-bar-open entry already erodes it to −0.047R gross,
and costs bury it to −0.18R net. Post-expansion bars also carry elevated
volatility, so a fixed 1.0×ATR-reference stop is hit more often than the 1.5R
target — WR 0.33–0.38 sits below the 0.40 breakeven. Continuation is not present
at an exploitable, after-cost horizon.

## Verdict

`STOP — volatility-expansion continuation has no directional edge` (Gate 3
failed: pooled after-cost non-positive; non-EURUSD negative; loses to
matched-random; not above shuffled-label at 95%). No robustness/validation gate
was opened. No parameter search or rescue was attempted. New research must start
from a separate hypothesis.

## Files

`config.py` (frozen params/splits/costs), `prepare_data.py` (loading + lockbox),
`signal.py` (frozen signal), `backtest.py` (single-entry after-cost R),
`baselines.py` (matched-random / shuffled / unconditional), `bootstrap.py`,
`run_gate1_audit.py`, `run_gate3.py`, `results/`, `test_registry.csv`.
