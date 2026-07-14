# Research log — diagnosing the NO-GO edge (branch `the5ers-bootcamp`)

Protocol: 12-month lockbox = **2025-03-20 → 2026-03-20** (end of data). All
diagnosis and iteration below uses only data **before 2025-03-20**. The lockbox
is run once, on the single final configuration, at the very end.
*Caveat recorded up front:* the original conversion session (2026-07-08) ran
the full history including 2025–2026 once with the baseline config, so the
lockbox is clean only with respect to configurations explored from here on.

Every configuration evaluated is logged here with fold results. Risk layer
(kill switch, sizing, limits) is frozen throughout.

---

## Phase 1 — data expansion (no rule changes)

- EURUSD M1 extended to 2015-01 → 2026-03 (HistData yearly archives;
  275,955 M15 bars, 0 duplicates, weekend/holiday gaps only).
- **Data defect found and fixed:** HistData timestamps are US Eastern local
  time, previously mislabelled as UTC → every prior backtest ran the London
  session filter ~5h off the true clock (commit 16eef3b).
- `data/news_events.csv` backfilled 2015→present: CPI (137 exact), FOMC
  (107 exact incl. unscheduled), ECB (92 decisions + pressers), NFP (36 exact
  2015-17 + first-Friday rule, 31/36 accuracy) (commit 8027451).

## Config #0 — BASELINE re-run (unchanged rules) on design window 2015-01 → 2025-03-19

Full window: **422 trades, PF 0.9898, WR 27.01%, avg R +0.029, return −0.95%,
maxDD 6.88%, longest streak 18, kill-switch hit: yes (research mode continues)**
MC (20k paths): P(pass)=0.304, P(kill)=0.696, P(breach −5%)=0.0000

| Fold | Test window | Trades | PF | WR% | avg R |
|---|---|---|---|---|---|
| 1 | 2017-01→2017-07 | 22 | 1.029 | 27.3 | +0.05 |
| 2 | 2017-07→2018-01 | 20 | 1.364 | 35.0 | +0.21 |
| 3 | 2018-01→2018-07 | 8 | 0.684 | 25.0 | −0.17 |
| 4 | 2018-07→2019-01 | 17 | 0.977 | 23.5 | +0.02 |
| 5 | 2019-01→2019-07 | 31 | 0.847 | 22.6 | −0.04 |
| 6 | 2019-07→2020-01 | 41 | 0.817 | 26.8 | −0.08 |
| 7 | 2020-01→2020-07 | 18 | 1.719 | 38.9 | +0.36 |
| 8 | 2020-07→2021-01 | 15 | 0.802 | 26.7 | −0.10 |
| 9 | 2021-01→2021-07 | 29 | 0.921 | 27.6 | −0.01 |
| 10 | 2021-07→2022-01 | 35 | 0.744 | 20.0 | −0.10 |
| 11 | 2022-01→2022-07 | 16 | 0.569 | 18.8 | −0.23 |
| 12 | 2022-07→2023-01 | 2 | 1.825 | 50.0 | +0.45 |
| 13 | 2023-01→2023-07 | 23 | 1.109 | 30.4 | +0.09 |
| 14 | 2023-07→2024-01 | 32 | 1.463 | 34.4 | +0.25 |
| 15 | 2024-01→2024-07 | 36 | 0.969 | 27.8 | +0.02 |
| 16 | 2024-07→2025-01 | 29 | 1.807 | 34.5 | +0.34 |
| 17 | 2025-01→2025-03 | 4 | 0.886 | 25.0 | −0.03 |

**Reading:** n=422 kills the "too small to conclude" problem. The strategy is
a coin flip minus costs: mean fold PF ≈ 1.01, 6/17 folds ≥ 1.0. Gross edge
per trade ≈ +0.12R before ~0.09R of cost drag. Diagnosis proceeds to Phase 2.

---

## Phase 2 — diagnostics (design window 2015-01 → 2025-03-19)

### 2a. Filter funnel (config #0, diagnostics mode; reproduces PF 0.9898 exactly)

| Stage | Count |
|---|---|
| bars | 250,979 |
| regime-valid bars (H1 EMA + ADX) | 115,046 (45.8%) |
| pullback-precondition bars | 250,959 (**~100% — vacuous**) |
| raw triggers (signals) | 5,358 |
| killed: session | 2,728 |
| killed: SL clamp | 1,271 |
| killed: position busy | 753 |
| killed: news / vol-floor / vol-ceiling / pacing | 91 / 75 / 9 / 9 |
| **executed** | **422** |

Shadow outcomes of killed candidates (R after costs): session −0.103 (n=1801),
sl_clamp (unclamped) −0.170 (n=1271), busy −0.129 (n=576), news −0.272 (n=14),
**vol_floor +0.238 (n=74 — the only filter removing profitable trades)**.
Conclusions: filters are fine (session/clamp/news all save money); the
pullback "touch" precondition is true on essentially every bar, so the entry
is really regime + RSI-recross.

### 2b. MAE/MFE (422 executed trades)

- 53.6% reach +1R (MFE median 1.07R, P75 2.01R); 27% reach the 2R TP.
- BE-stopped: 109 (25.8%); of those 29.4% would have reached 2R without BE.
- **BE at 1R is net positive**: actual avg R +0.029 vs no-BE counterfactual
  −0.033 (+0.06R/trade saved). Exit logic is NOT the culprit. Only 1 of 199
  full losers had MFE > 1.5R.

### 2c. Component ablation (same exits/costs/session, no pacing)

| Variant | n | PF | Expectancy | WR |
|---|---|---|---|---|
| (a) regime-only, first in-session bar/day | 1128 | **1.118** | **+0.064R** | 31.5% |
| (b) regime + limit at EMA20 zone | 1583 | 0.701 | −0.190R | 22.7% |
| (c) full trigger (pullback+RSI recross) | 468 | 0.988 | −0.006R | 26.9% |

Per-fold: (a) positive in 12/21 half-years (strong 2015-18, soft 2019-21);
(b) negative nearly everywhere; (c) noise around 1.0.

**Diagnosis:** the H1 regime carries a real edge; the M15 confirmation
trigger subtracts it ((c) < (a)); entering the pullback early via limit
order is much worse ((b) ≪ (a)) — with 8–25-pip stops, catching the
retracement knife gets stopped before the trend resumes. The evidence
indicts the TRIGGER component, not exits, not filters (except vol-floor,
n=74, weakly indicted).

---

## Phase 3 — targeted changes (design window only; one at a time)

Ruled out without a config run (funnel evidence): widening the ATR floor —
in entry-anchored-SL mode every ATR < 5.33 pips candidate dies at the frozen
8-pip SL clamp anyway (vol_floor kills have zero shadow survivors). The SL
clamp itself is frozen risk policy. BE/TP already shown net-positive in 2b.

## Config #1 — entry_mode=regime_daily, 08:00-bar-only (WRONG implementation)
Sampled ONLY the day's first session bar; no retry if regime absent/clamp
failed at 08:00. **trades=634 PF 0.778, 0/17 folds ≥ 1.25. Rejected** —
and identified as an unfaithful port of ablation (a).

## Config #1b — regime_daily, retry all in-session bars (1 entry/day)
**trades=1620 PF 1.005**, 4/17 folds ≥ 1.0, 2/17 ≥ 1.25. Later-in-day
entries dilute the edge. Rejected.

## Config #1c — regime_daily, first regime-valid bar per day, no retries
Faithful port of ablation (a) under full production gates/costs.
**trades=1106, PF 1.100, WR 31.1%, avg R +0.118 (gross of commission),
return +18.6%, maxDD 9.33%, MC P(pass)=0.559 / P(kill)=0.441 / P(breach)=0.**
Folds: 3/17 ≥ 1.25, 5/17 ≥ 1.0 (min 0.626, median 0.915).
MAE/MFE: 56% reach 1R, 40% reach 1.5R, 31% hit 2R TP; BE-at-1R again net
positive (+0.12 vs +0.02 no-BE). **Best candidate so far.**

## Configs #2–#9 — exit sweep on #1c (BE ∈ {1.0, 1.25, 1.5, none} × TP ∈ {2.0, 1.5})

| BE | TP | overall PF | folds ≥1.0 | folds ≥1.25 | min PF | median PF |
|---|---|---|---|---|---|---|
| 1.0 | 2.0 | **1.100** | 5 | **3** | 0.626 | 0.915 |
| 1.0 | 1.5 | 1.075 | 7 | 3 | 0.745 | 0.958 |
| 1.25 | 2.0 | 1.107 | 7 | 3 | 0.498 | 0.932 |
| 1.25 | 1.5 | 1.080 | 7 | 1 | 0.608 | 0.935 |
| 1.5 | 2.0 | 1.059 | 8 | 2 | 0.475 | 0.932 |
| 1.5 | 1.5 | 1.045 | 5 | 2 | 0.579 | 0.953 |
| none | 2.0 | 1.031 | 6 | 1 | 0.520 | 0.907 |
| none | 1.5 | 1.047 | 5 | 1 | 0.579 | 0.953 |

Flat surface (±0.05 PF): exits are NOT the lever. Existing BE 1.0 / TP 2.0
kept (top of grid, and no basis to switch on noise-level differences).

Regime alternates (H4 EMA, Donchian slope) NOT explored: the protocol only
authorizes them if the regime is uninformative — it is informative
(ablation (a) beats random and carries all the edge). MetaVeto test NOT
run: pre-condition "rules stream ≥ breakeven on folds" is not met (5/17).

**FINAL CONFIGURATION (frozen before lockbox): config #1c** —
entry_mode=regime_daily, all other production parameters unchanged
(SL 1.5×ATR entry-anchored clamped 8–25 pips, TP 2R, BE 1R, 0.3% risk,
1 trade/day in this mode, all filters, all pacing stops).
Search size disclosure: 12 configurations evaluated on the design window
(#0, #1, #1b, #1c, 8 exit variants); zero lockbox contact so far.
