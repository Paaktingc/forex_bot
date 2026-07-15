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

---

## Phase 4 — final validation (single pass, config #1c frozen)

**Lockbox (2025-03-20 → 2026-03-20, opened exactly once):**
111 trades, PF 0.894, WR 26.1%, avg R +0.005, return −2.02%, maxDD 4.97%.
Lockbox-vs-folds check: PF 0.894 vs design fold median 0.915 → within the
pre-registered 0.2 band (consistent, i.e. honestly *flat*, not degraded).

**Official `python backtest.py --go-no-go` (full 2015→2026 history):**
1217 trades, PF 1.081, WR 30.7%, avg R +0.108 (gross), return +16.2%,
maxDD 11.4%, kill switch never hit on the realized path.
Walk-forward: 4/19 folds PF ≥ 1.25 (perturbation bands wide, several folds
robustly < 1.0). MC (20k paths): P(pass)=51.4%, P(kill)=48.6%,
P(breach −5%)=0.00%, median 129 trades to pass.

| Gate | Required | Result | |
|---|---|---|---|
| P(breach −5%) | < 1% | 0.00% | PASS |
| P(kill switch −3%) | < 10% | 48.6% | FAIL |
| P(pass +6%) | > 70% | 51.4% | FAIL |
| PF ≥ 1.25 after costs, every fold | 19/19 | 4/19 | FAIL |
| Lockbox PF within 0.2 of folds | ≤ 0.2 | 0.02 | PASS |

## VERDICT: NO-GO (final; no further iteration against the lockbox)

The diagnosis is complete and the answer is clear: risk containment is
proven (the official −5% is unreachable), the H1 regime carries a real but
small edge (~+0.11R gross, ~+0.01–0.04R net of costs), and no component of
the pullback framework — trigger, exits, filters — can amplify it to the
gate level. The edge/cost ratio of a single-instrument EURUSD M15 system at
8–25-pip stops is the binding constraint.

**Recommended next research direction (not implemented here):**
1. Multi-instrument pooling (project Lever 1): the regime_daily entry is
   mechanical and symbol-agnostic; pooling 4–6 uncorrelated majors at the
   same 0.3% risk multiplies trade count and diversifies fold variance
   without touching per-trade risk. GBPUSD/USDJPY/AUDUSD M1 data already in
   m1_data/.
2. Reduce cost drag structurally: H1-native entries (fewer, wider stops
   within the 25-pip clamp) cut the spread+commission share of R.
3. Only after (1)/(2): revisit the gates with the pooled trade stream.

---
---

# Research cycle 2 (2026-07-15) — pursuing the two recommended directions

User-authorized continuation targeting the failing gates. Pre-registration:

- **Gates unchanged**, risk layer unchanged (kill switch, 0.3% sizing,
  8–25-pip SL clamp, pacing all frozen).
- Design window unchanged: data before **2025-03-20**. Lockbox unchanged:
  2025-03-20 → end. *Disclosure:* the EURUSD lockbox was opened once in
  cycle 1 (config #1c, PF 0.894); it is partially burned for EURUSD and
  fresh for other instruments. It will be opened once more, only for the
  single final cycle-2 configuration.
- **Lever A (cost drag):** widen the SL ATR multiplier (a strategy param
  already perturbed in walk-forward, NOT frozen risk) within the frozen
  clamp: sl_atr_mult ∈ {1.5 base, 2.0, 2.5, 3.0}. Rationale: costs are
  ~1.4–1.9 pips per trade; on a 12-pip stop that is ~0.13R, on a 22-pip
  stop ~0.07R — the diagnosis showed the gross edge (+0.11R) is real but
  cost-consumed. Exits stay TP 2R / BE 1R (proven). Selection rule
  (pre-registered): pick by fold consistency (folds ≥ 1.0, then min PF),
  not best average; ties → smaller change from base.
- **Lever B (pooling, project Lever 1):** run the SAME regime_daily rules
  on GBPUSD, USDJPY, AUDUSD (2015→design end), per-pair conservative
  spread floors, same commission; pool trade streams under the live
  constraint of ONE open trade globally (first signal wins, ties by
  symbol alphabetical — deterministic). Selection rule: pooling is adopted
  only if it improves fold consistency vs the best single-instrument
  config without raising per-trade risk.
- Every configuration logged. Final config → lockbox once → gates.

## Cycle-2 configs #10–#12 — Lever A: sl_atr_mult sweep (EURUSD design window)

| sl×ATR | trades | overall PF | avg R | maxDD | folds ≥1.0 | ≥1.25 | min | median |
|---|---|---|---|---|---|---|---|---|
| 1.5 (base) | 1106 | **1.100** | +0.118 | 9.3% | 5 | 3 | 0.626 | 0.915 |
| 2.0 | 1408 | 1.000 | +0.056 | 21.6% | 6 | 1 | 0.699 | 0.916 |
| 2.5 | 1255 | 0.927 | +0.004 | 24.8% | 5 | 0 | 0.581 | 0.906 |
| 3.0 | 1101 | 0.915 | −0.009 | 22.1% | 3 | 0 | 0.567 | 0.849 |

**Lever A REJECTED** — monotonically worse: the cost-share saving of wider
stops is dominated by the win-rate collapse as the 2R target moves further.
Base 1.5×ATR retained. Cycle 2 proceeds on Lever B (pooling) only.

## Cycle-2 configs #13–#18 — Lever B: per-pair runs and pooling (design window)

Per-pair regime_daily under identical production gates (per-pair spread
floors 0.6/0.9/0.7/0.8 pips, same commission):

| Symbol | n | PF | avg R | ret | maxDD |
|---|---|---|---|---|---|
| EURUSD | 1106 | 1.100 | +0.118 | +18.6% | 9.3% |
| GBPUSD | 1536 | **1.112** | +0.118 | +30.2% | 8.5% |
| USDJPY | 1118 | 0.952 | +0.058 | −9.8% | 19.3% |
| AUDUSD | 1092 | 0.939 | +0.029 | −11.4% | 23.7% |

**GBPUSD independently replicates the regime edge on a fresh instrument**
(same rules, no tuning). USDJPY/AUDUSD do not carry it — the London
first-bar entry is mistimed for Asia-driven pairs. Known limitation: BoE/
BoJ/RBA event blackouts are absent from news_events.csv (USD events cover
all pairs; EUR covers EURUSD) — flagged, would only worsen UJ/AU further.

Pooled under live constraints (ONE open trade globally, global pacing,
0.3%/trade on pooled balance):

| Pool | n | PF | folds ≥1.0 | ≥1.25 | min | median | MC P(pass) | P(kill) |
|---|---|---|---|---|---|---|---|---|
| all 4 | 2861 | 1.048 | 8/17 | 2/17 | 0.613 | 0.995 | 45.6% | 54.4% |
| EU+GU | 2150 | 1.086 | **9/17** | 3/17 | 0.660 | **1.002** | 51.5% | 48.5% |

Pooling improves fold consistency (median PF 1.00 vs 0.915 single-pair) but
LOWERS overall PF vs single-pair EU/GU: the one-open-trade constraint makes
correlated signals displace each other. Selection per pre-registered rule
(fold consistency): **EU+GU pool** is the best cycle-2 configuration.

## Cycle-2 verdict — NO-GO, lockbox NOT opened (protocol deviation, documented)

The best configuration fails the gates on design data alone: 3/17 folds
≥ 1.25 (need all), MC P(pass)=51.5% (need >70%), P(kill)=48.5% (need <10%).
Opening the lockbox cannot change the verdict and would burn the holdout
for a config that already failed — deviating from the pre-registered
"lockbox once" step FOR THAT REASON, the lockbox stays sealed for cycle 2
(still only ever opened once, in cycle 1, for EURUSD config #1c).

Search size: cycle 2 evaluated 9 further configurations (3 SL widths beyond
base, 4 per-pair runs, 2 pools). Cumulative across both cycles: 21 configs,
all logged here.

### Where this leaves the project

Consistent evidence across 21 configurations, 4 instruments, 10 years:
the H1-regime/London-morning entry has a REAL but SMALL edge
(≈ +0.10–0.12R gross, ≈ +0.02–0.05R net of realistic costs), replicated
out-of-family on GBPUSD. The Bootcamp gates need roughly +0.25R net.
This framework cannot bridge that gap by configuration; it needs either
(a) structurally lower costs (better broker terms would nearly double net
expectancy — the strategy is cost-bound, not signal-bound), or
(b) a second, uncorrelated signal family (e.g. non-London sessions or
mean-reversion regime complement) to raise pooled expectancy without
displacement, or (c) accepting that a 50% step-pass probability with zero
breach risk is simply what this edge is worth — below the bar for a
funded-account attempt.
