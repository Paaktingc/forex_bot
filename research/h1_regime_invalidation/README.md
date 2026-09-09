# H1 Regime Strategy — Invalidation by Higher-Timeframe Look-Ahead Leakage

## Executive summary

The H1 regime strategy's historically reported positive expectancy was **caused
by look-ahead leakage**, not by an edge. The strategy read the eventual close of
a *still-forming* H1 bar when making M15 entry decisions inside that H1 hour.

After correcting the alignment to use only the most recently **closed** H1 bar,
expectancy is materially **negative** across all four available FX pairs:

| Pair | Honest E[R] | PF | iid 95% CI | (Leaky E[R]) |
|---|---|---|---|---|
| EURUSD | **−0.1295R** | 0.794 | [−0.205, −0.054] | +0.0965R |
| GBPUSD | **−0.1038R** | 0.828 | [−0.168, −0.040] | +0.1132R |
| AUDUSD | **−0.1515R** | 0.753 | [−0.225, −0.077] | +0.0224R |
| USDJPY | **−0.1190R** | 0.807 | [−0.192, −0.044] | +0.0241R |

Every honest 95% confidence interval is entirely below zero. **The strategy is
retired.** No rescue or optimisation was attempted.

## Technical root cause

- **Left-labelled H1 bars.** The `*_H1_real.csv` bars (and `resampler.resample_ohlcv`,
  `label="left", closed="left"`) are labelled by **open** time. A bar stamped
  `09:00` covers `[09:00, 10:00)` and does not **close** until `10:00`. Verified:
  the H1 bar's open equals the first constituent M15 open and its close equals
  the last constituent M15 close.
- **Feature availability.** Any H1 feature (EMA, ADX, regime = `close ≷ EMA`) is
  known only at the bar's **close** = `label + 1h`, never at the opening label.
- **The leak.** `strategy.build_signal_frame` aligned the H1 regime onto M15 with
  `pd.merge_asof(direction="backward")` keyed on the H1 **label**. An M15 bar at
  `09:00 / 09:15 / 09:30 / 09:45` was therefore matched to the H1 bar labelled
  `09:00` — still forming, closing at `10:00`. The regime it read depended on
  the H1 bar's own future close.
  - Confirmed instance: M15 `09:15` → H1 label `09:00` (closes `10:00`). The last
    H1 bar actually closed by `09:15` is the `08:00` bar.
- **The fix.** `htf_alignment.align_last_closed_bar` computes an explicit
  availability timestamp `available_at = open + timeframe` and aligns with
  `merge_asof` on availability (`allow_exact_matches=True`, so a bar closing
  exactly at `t` is usable at `t`). Invariant:

  > At an entry decision timestamp `t`, the strategy may use only an H1 bar whose
  > close timestamp is ≤ `t`.

  Wired into `strategy.build_signal_frame` and `features.add_h1_trend`.

## Reproduction

- Honest baseline + entry-set decomposition: `baseline_honest.py`
  (output snapshot: `baseline_honest.txt`).
- Gate 0 diagnosis (leaky vs honest, per-instrument): `../gate0_reconcile.py`,
  `../gate0_leakage_probe.py` (commit `9b49079`).

**Entry-set decomposition (EURUSD, Window A, leaky vs honest):**

| Group | n | E[R] |
|---|---|---|
| leak-only (intra-hour, forming-bar) | 543 | **+0.307R** (the false edge) |
| common to both alignments | 585 | −0.099R |
| honest-only | 523 | −0.164R |

The entire positive expectancy lived in the intra-hour leak-only entries whose
regime label already "knew" how their H1 bar would close. Leaky counts per pair:
EURUSD 1128, GBPUSD 1547, AUDUSD 1104, USDJPY 1138; leak-only ≈ 543/626/432/429
with E[R] +0.307/+0.482/+0.194/+0.287.

**EURUSD windows (honest):** Window A 2015-01→2025-03 n=1108, E[R] −0.1295,
1/11 years positive; Window B 2007→2025 n=2210, E[R] −0.0917, 4/19 positive.

**Cost reconciliation (unaffected, stands).** The previously-reconciled
+0.0644R (0.6-pip/$7) vs +0.0965R (0.4-pip/$4) gap is entirely cost, on the
identical 1,128 leaky entries: commission $7→$4 = +0.0268R (83.5%), spread
0.6→0.4 = +0.0053R (16.5%), additive. This is **not** the leakage discrepancy.

## Leakage audit

Whole-repo scan for `merge_asof`, `resample`, `ffill`/forward-fill, `shift(-1)`,
centred rolling windows, and higher-timeframe→lower-timeframe joins.

| Location | Feature / timeframe | Old alignment | Available at | Leak | Fix |
|---|---|---|---|---|---|
| `strategy.build_signal_frame` | H1 regime + H1 swings → M15 | `merge_asof` backward on H1 **label** | H1 close = open+1h | **YES (confirmed)** | ✅ `align_last_closed_bar` |
| `features.add_h1_trend` | H1 close/EMA50 → M15 (ML matrix) | `merge_asof` backward on H1 **label** | H1 close = open+1h | **YES (confirmed)** | ✅ `align_last_closed_bar` |
| `resampler.resample_ohlcv` | M1→H1 aggregation | `resample(label="left", closed="left")` | bar close = label+tf | No (correct resample; it is the *source* of left-labelling, consumers must respect it) | none |
| `prepare_data.py:107` | data-prep resample | `df.resample(timeframe)` | — | No (offline data build) | none |
| `features.compute_regime_indicators_h1` / `compute_entry_indicators_m15` | EMA/ADX/ATR/RSI | trailing `ewm`/rolling on same frame | bar close | No (causal, same-frame) | none |
| `features.rolling_swing_levels` | H1 swing hi/lo | trailing `rolling().max/min` | bar close | No (causal; leak was only in the *alignment*, now fixed) | none |
| `validation.py:271` | daily returns of realised trades | `resample("1D")` | post-trade | No (analytics on closed trades) | none |
| `_ema` / `_atr` / `_rsi_wilder` / `_adx` | indicators | trailing only; no `shift(-1)`, no `center=True` | bar close | No | none |

Two confirmed leaks (same structural class), both fixed. No `shift(-1)`, no
centred windows, and no future-outcome columns found in feature construction.

## Impact assessment

- **Invalidated research** (used `build_signal_frame`'s regime): design-window
  and extended-window results, the Phase-4 lockbox, the Phase-2c ablation, all
  configs #1–#24, cycles 2–6, Step 1 feasibility, Gate A. See the invalidation
  banner/table at the top of `research_log.md`.
- **Invalid Monte Carlo:** every `monte_carlo_dd` output built on the above
  return streams (P(pass)/P(kill)/maxDD, feasibility envelopes). The MC code is
  correct; its **input return distribution** was a leakage artifact.
- **Invalid programme recommendations:** any Bootcamp/High-Stakes recommendation
  derived from those MC outputs.
- **Affected production path:** the live loop (`main.process_candle` →
  `strategy.generate_candidate` → `build_signal_frame`) with the default
  `ENTRY_MODE="regime_daily"`. Now blocked at startup by the retirement guard.
- **Orders historically generated?** No live/demo order history is present in
  this repository, so none can be attributed here. (Stated as fact about the
  repo, not an inference about any live account.)
- **Not affected:** the Gate 0 XAUUSD-copier stop-loss counterfactual (separate
  line), and the cost reconciliation.

## Prevention

- **Mandatory tests:** `tests/test_htf_alignment.py` (last-closed alignment,
  exact-close boundary, no-feature-before-first-close, missing bars, tz-aware,
  DST, H4 inference, trailing-window/no-negative-shift guards, `add_h1_trend`
  leak check); `tests/test_regime_alignment_leakage.py` (production alignment
  invariant on real data — fails if the leaky label-merge returns);
  `tests/test_strategy_retirement.py` (retired modes cannot go live).
- **Shared utility:** `htf_alignment.align_last_closed_bar` — all
  higher-timeframe→lower-timeframe joins must go through it (or an explicit
  close-timestamp), never a backward `merge_asof` on an opening label.
- **Research/live parity:** the live signal factory and the research/backtest
  path share the single fixed `build_signal_frame`, so alignment cannot diverge.
- **Review rule for future strategies:** every feature must have a stated
  *availability timestamp* ≤ the decision timestamp; left- vs right-labelled bar
  semantics must be explicit before any cross-timeframe join.
