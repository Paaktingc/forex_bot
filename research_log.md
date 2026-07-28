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

---

# Research cycle 3 (2026-07-15) — cost sensitivity + second session anchor

Pre-registration:
- **Lever C (cost sensitivity, no config change):** re-run the EU+GU pooled
  configuration under 4 cost models — conservative (current: 0.6/0.9-pip
  floors, $7/lot RT, 0.3-pip entry slip), typical raw-ECN (0.2/0.3, $6,
  0.2), premium (0.1/0.2, $5, 0.1), zero (upper bound). Purpose: measure
  how much of the gate shortfall is execution cost vs signal. This changes
  NO conclusions by itself — the production cost model stays conservative
  unless real The5ers conditions are verified better by the user.
- **Lever D (second anchor within the framework):** entry_mode
  "regime_daily2" = regime_daily plus ONE additional entry at the first
  regime-valid bar at/after 13:00 London (NY-open overlap), max 2/day
  global (existing cap), all other rules identical. Rationale: different
  time-of-day anchor should reduce cross-pair displacement in the pool.
  Selection rule unchanged: fold consistency on the design window.
- Lockbox remains sealed unless a cycle-3 config passes all gates on
  design data (same rule as cycle 2).

## Cycle-3 Lever C — cost sensitivity (EU+GU pool, design window)

| Cost model | PF | folds ≥1.0 | ≥1.25 | median PF | P(pass) | P(kill) |
|---|---|---|---|---|---|---|
| conservative (prod: 0.6/0.9p, $7, 0.3 slip) | 1.086 | 9/17 | 3/17 | 1.002 | 51.5% | 48.5% |
| typical raw ECN (0.2/0.3p, $6, 0.2) | 1.191 | 12/17 | 7/17 | 1.223 | 70.4% | 29.6% |
| premium (0.1/0.2p, $5, 0.1) | 1.256 | 11/17 | 8/17 | 1.248 | 78.4% | 21.6% |
| zero (upper bound) | 1.428 | 15/17 | 11/17 | 1.479 | 90.0% | 10.0% |

Execution costs are worth ≈0.34 PF. Under typical raw-ECN terms P(pass)
reaches the 70% gate — but P(kill) (30%) and the every-fold PF gate (7/17)
still fail, and **even at ZERO cost the every-fold gate fails (11/17)**:
beyond costs, the edge's time-variability independently blocks the gates.
Production cost model stays conservative (no config change).

## Cycle-3 config #22 — Lever D: regime_daily2 (second NY-open anchor) — REJECTED

EURUSD n=1991 PF 0.930; GBPUSD n=2454 PF 1.004; EU+GU pool n=3132
**PF 0.941**, folds ≥1.0: 4/17, P(pass)=21.2%. Attribution: the
afternoon-entry slice alone is NEGATIVE (n=1402, PF 0.928, −0.04R net).
The edge is specifically the London-morning regime continuation; every
other time-of-day variant tested (afternoon anchor, all-day retries,
zone limits) subtracts value.

## Cycle-3 verdict — NO-GO stands; lockbox still sealed; framework exhausted

22 configurations across 3 cycles. Stable conclusions:
1. The London-morning H1-regime edge is real (+0.10–0.12R gross,
   replicated untuned on GBPUSD) and fully characterized.
2. It is cost-bound first (≈0.34 PF of drag) and consistency-bound second
   (even costless, per-fold PF ≥ 1.25 everywhere is out of reach).
3. No configuration of this framework can pass the Bootcamp gates.
   Further tuning would be curve-fitting; stopping per protocol.

Actionable residuals for the user (outside backtest scope):
- Verify The5ers' ACTUAL Bootcamp spreads/commissions; if they are at
  raw-ECN levels, the measured P(pass) is ≈70% (still short on P(kill)
  and fold consistency — informational, not a GO).
- A genuinely new, uncorrelated signal family (different session/style)
  is the only remaining path to the gates; that is new-strategy research,
  not refinement of this one.

---

# Research cycle 4 (2026-07-15) — verified costs + second signal family

Goal (user-directed): build the strongest legitimate case for a GO. The
gates are unchanged and will not be bent; if the evidence falls short the
verdict stays NO-GO.

Pre-registration:
- **Verified execution costs (primary source):** The5ers Help Center
  "What are the spreads and commissions?" (updated 02.01.2026): majors
  "sell from 0.2 pips to 0.9 pips"; forex commission **$4/lot round trip**.
  Production backtest cost model updated to verified-conservative:
  spread floors EURUSD 0.4 / GBPUSD 0.6 pips (upper-middle of the quoted
  range), commission $4/lot RT. Slippage assumptions UNCHANGED (0.3 entry
  / 1.0 stop / 2.0 news — not covered by their terms, stays ours,
  conservative). Cycle-3 pre-registration explicitly conditioned this
  update on primary-source verification; done.
- **Family 2 — "range_fade" (fully specified before any run, 1 variant):**
  On days with NO H1 regime at the prior close (regime==0 — orthogonal to
  family 1 by construction): Asian range = high/low of 00:00–07:59 London
  (≥12 bars required). In the 08:00–10:59 London entry window, if a bar
  pokes above the Asian high and closes back inside → SHORT at next open;
  mirror below the low → LONG. SL = 1.5×ATR beyond the poke extreme
  (existing swing/clamp machinery, frozen 8–25-pip clamp), TP 2R, BE 1R
  (identical exits to family 1), one trade/day/symbol, all existing
  filters and pacing. NO parameter search: k=0 poke, fixed windows.
- **Pooling:** family1+family2 on EU+GU (4 streams) under live global
  constraints (one open trade, 2 entries/day global). Gates evaluated on
  the design window; the lockbox is opened ONLY if all gates pass there.

## Cycle-4 results

**Config #23 — family 2 "range_fade" solo (design window): REJECTED.**
EURUSD n=788 PF 0.853 (−0.06R); GBPUSD n=682 PF 0.798 (−0.10R). The
failed-breakout fade of the Asian range on no-regime days has NO edge with
2R targets through 8–25-pip stops. Per pre-registration (single variant, no
parameter search) it is rejected, not tuned; pooling a negative stream is
pointless. Mean reversion joins the list of components that subtract value.

**Config #24 — family-1 EU+GU pool under VERIFIED The5ers costs
(0.4/0.6-pip floors, $4/lot RT, slippage unchanged):**
n=2150, **PF 1.165, WR 31.0%, +69.8% over 8.2y, maxDD 9.96%**
Folds: 11/17 ≥ 1.0, 5/17 ≥ 1.25, min 0.727, median 1.127
MC (20k): **P(pass)=66.4%, P(kill)=33.6%, P(breach −5%)=0.00%**, median
112 trades to pass.

| Gate | Required | Result | |
|---|---|---|---|
| P(breach −5%) | < 1% | 0.00% | PASS |
| P(kill −3%) | < 10% | 33.6% | FAIL |
| P(pass +6%) | > 70% | 66.4% | FAIL |
| PF ≥ 1.25 every fold | 17/17 | 5/17 | FAIL |

## Cycle-4 verdict — NO-GO stands. Lockbox still sealed.

24 configurations, 4 cycles. The verified-cost update moved P(pass) from
51.5% → 66.4% — the single biggest legitimate improvement found — but the
gates require an edge this framework does not have. Both tested candidate
families for a consistency-fixing second stream (afternoon momentum, Asian-
range fade) are negative. Continuing to generate new families ad hoc would
be data mining: each additional test cheapens any future "pass".

Honest economics of the best configuration (informational, NOT a GO):
P(pass one step) ≈ 0.66 → P(pass 3 Bootcamp steps) ≈ 0.66³ ≈ **29%**, with
~34% chance per step of ending at the −3% self-imposed halt and ~0% chance
of breaching the official −5%. Expected outcome of a challenge fee at these
numbers is negative. The GO bar (P(pass)>70% per step, every-fold PF≥1.25)
remains the right bar, and this strategy does not meet it.

---
---

# Research cycle 5 (2026-07-16) — multi-instrument transfer test (user plan)

Adopting the user-supplied test plan with two honest amendments, all frozen
BEFORE any backtest run:

- **Universe & priority:** GRXEUR (DAX proxy) and ETXEUR (EURO STOXX 50
  proxy) first — session alignment + The5ers indices are commission-free;
  XAUUSD second (real London prior, metal costs assumed pending MT5 spec
  freeze); **US30 DEFERRED** — HistData lists no Dow proxy and the clean
  source (CME DataMine) is unavailable in this environment; running it on
  a degraded substitute would be data theater. EU/GU history extension to
  2007–2014 (genuinely unseen years) included as the cheapest evidence.
- **Fixed-rule transfer:** identical rules, London clock, exits (TP 2R,
  BE 1R), filters, pacing. NO re-anchoring to local opens, NO per-symbol
  threshold edits.
- **Unit mapping (pre-registered, one rule for all non-FX):** pip-based
  thresholds convert to basis points of price calibrated once from
  EURUSD@1.10 — SL clamp 8–25 pips → 7.27–22.73 bp; ATR floor 4 pips →
  3.64 bp; slippage 0.3/1.0/2.0 pips → 0.273/0.909/1.818 bp. Applied at
  each trade's entry price. FX symbols keep the existing pip rules
  unchanged.
- **Costs (instrument-specific, per The5ers help centre):** indices —
  ZERO commission, spread floors ASSUMED (GRXEUR 1.5 pts, ETXEUR 1.5 pts)
  pending MT5 spec freeze; XAUUSD — percentage commission ASSUMED at
  0.002%/side (≈$8/lot RT at $2000) pending MT5 spec freeze; FX unchanged
  (verified). Any GO-relevant result must be re-frozen against live MT5
  specs before being believed.
- **Decision unit = the POOL.** Solo tables are diagnostics only. Augmented
  pools tested: EU+GU+{GRXEUR}, +{ETXEUR}, +{XAUUSD}, then best-two if any
  single addition helps. Uplift assessed with a PAIRED day-level moving-
  block bootstrap (same resampled days for baseline and augmented pool,
  same seeds) reporting CIs for ΔPF and ΔP(pass). An uplift whose CI
  straddles zero is not a flip candidate.
- Gates unchanged. Lockbox (2025-03-20→) stays sealed unless a pooled
  config passes ALL gates on design data.

## Cycle-5 results

**Data notes (all disclosed):** ETXEUR DROPPED — HistData feed degrades
2017→2019 and dies after Feb-2020 (16k bars in 2019). GRXEUR feed is
contaminated with EURO STOXX-scale prices 2020-06-15→2023-12-03 (provider
switch); only the verified DAX-scale windows are used (clean-window filter
in prepare_data, 637,761 M1 rows dropped). XAUUSD clean. EU/GU extended to
2007. US30 deferred (no credible source available).

### Solo transfer tests (design window, fixed rules, class-correct costs)

| Test | n | PF | folds ≥1.0 | median PF | Verdict |
|---|---|---|---|---|---|
| **EURUSD 2007–2014 (8 UNSEEN years)** | 1100 | **1.324** | 12/16 | 1.236 | strongest validation yet |
| **GBPUSD 2007–2014 (UNSEEN)** | 1023 | 1.167 | 12/15 | 1.162 | replicates |
| **GRXEUR (DAX, clean segments)** | 666 | 1.166 | 11/15 | 1.130 | transfers (3rd instrument) |
| XAUUSD | 1489 | 0.950 | 10/21 | 0.977 | no transfer — rejected |

### Decisive stage: pooled comparison (design window, paired day-bootstrap)

| Pool | n | PF | folds ≥1.25 | min | med | P(pass) | P(kill) |
|---|---|---|---|---|---|---|---|
| BASE EU+GU | 2150 | 1.1645 | 8/21 | 0.727 | 1.147 | 66.4% | 33.6% |
| AUG EU+GU+DAX | 2501 | 1.1576 | 9/21 | 0.803 | 1.175 | 67.2% | 32.8% |

Paired moving-block day bootstrap (3000 draws, 5-day blocks, same days both
pools): **ΔPF −0.007, CI95 [−0.054, +0.039]; ΔP(pass) −1.8pp.** The CI
straddles zero → per the pre-registered rule, DAX addition is NOT adopted.
Mechanism: all three instruments enter at the same London-morning slot
under the one-open-trade constraint — they displace, not diversify.

## Cycle-5 verdict

The edge is now validated beyond reasonable doubt: same untouched rules
profitable on THREE instruments and on EIGHT years of data that did not
exist during design (2007–2014 EURUSD PF 1.32). But the Bootcamp gate
math is unchanged: P(pass) ≈ 66–67% (gate >70%), P(kill) ≈ 33% (gate
<10%), every-fold PF ≥ 1.25 unreachable (8–9 of 21). **NO-GO under the
frozen gates. Lockbox still sealed since cycle 1.**

The binding constraint is now provably NOT the signal, NOT the costs, and
NOT the instrument set — it is the strategy's ~one-good-trade-per-morning
capacity versus a three-step, +6%-per-step gauntlet. 30 configurations
across 5 cycles all land within a few points of the same step probability.
That number is the truth about this strategy.

# Research cycle 6 (2026-07-21) — programme-fit pivot: Bootcamp → High Stakes

Deep-research report: `cycle6_research_report.md` (18 sections, verified
The5ers rules + literature review + sleeve Monte Carlo). Headline: the
NO-GO is a **structure** problem, not an alpha problem. Do NOT tune the
frozen London rules further.

## Key finding — the barrier geometry, not the edge, is the constraint

Bootcamp is an asymmetric +6% / −3%-kill game with a thin (~+0.11R/trade)
edge. With unlimited time (VERIFIED), pass probability is governed by
edge/variance vs the barriers — not total return. Two levers move it:

1. **Lower risk** (frozen system only, MODEL): 0.30%→P(pass)~67%; 0.20%
   ~79%; 0.15%~86%; 0.10%~95% — but 430–770 trading days/stage. Raising
   risk strictly HURTS (0.60%→52%).
2. **Wider symmetric barrier** = a different programme.

## High Stakes fits the frozen edge far better (VERIFIED rules 2026-07-21)

2-step, targets 10% then 5%, **static 10%** max loss, 5% daily, ≥3
profitable days (day closing ≥0.5%), unlimited time, 1:100. A 10%/10%
symmetric barrier suits a thin positive edge. MODEL (calibrated frozen
ledger, block-bootstrap through the new programme simulator):

| Programme | Geometry | Risk | P(complete) |
|---|---|---|---|
| Bootcamp | 3×(+6) / −5 / −3 kill | 0.30% | ~0.30–0.36 |
| **High Stakes** | +10,+5 / −10 / −6 kill | 0.40% | **~0.72** |

Per-step High Stakes: step1 P(pass)~0.83 P(kill)~0.17; step2 ~0.86/0.14;
worst-step kill well under the Bootcamp ~0.29. Same untouched edge.

## Implemented this cycle (code)

- `config.py`: **PROGRAMMES** profile layer (`bootcamp`, `high_stakes`),
  selected by `PROGRAMME` env var. All barrier/kill/pacing/risk constants
  now DERIVE from the active profile (backward-compatible aliases kept).
  High Stakes circuit breakers retuned to the 10% budget: soft-reduce
  (halve size) at −4%, kill (flatten+disable) at −6%, official backstop
  −10%; daily pacing 1.5% (inside the official 5%); risk 0.40%;
  MIN_PROFITABLE_DAYS=3 (profitable day = closed ≥0.5%).
- `monte_carlo_dd.py`: `run_programme_monte_carlo()` +
  `run_programme_from_config()` + `print_programme_report()` — chains N
  steps with per-step geometry, reports per-step P(pass/kill/breach) and
  full-programme completion. Single-step API unchanged.
- `risk_manager.py`: profitable-day crediting at each server-day reset,
  `profitable_days_met()`, and `check_official_daily_loss()` backstop
  (no-op for Bootcamp). Kill switch / max-loss already followed the config
  constants, so they became programme-aware for free.
- `backtest.py --go-no-go`: now also prints the full-programme report and
  bases the verdict on the WORST per-step probabilities of the active
  profile.
- `tests/test_programme_profiles.py`: 9 tests (profiles, MC step math,
  HS>Bootcamp on identical edge, zero official breach, profitable-day gate,
  official-daily-loss backstop). Full suite: 233 passed (remaining
  failures/errors are missing-XGBoost / sandbox-network only).

## How to run

```
# Bootcamp (default) vs High Stakes go/no-go on the frozen strategy:
python backtest.py --go-no-go
PROGRAMME=high_stakes python backtest.py --go-no-go
```

## Cycle-6 verdict

**Switch the frozen London system to The5ers High Stakes $100K at
0.30–0.40% risk** (circuit breaker −6%/−8%). Do NOT run Bootcamp at 0.30%.
Do NOT tune London parameters. Parallel research track (optional): Cycle-6
NY-ORB-NAS100 sleeve, pre-registered v1.0 in `cycle6_research_report.md`
§10 — P(GO) ~25–35%; needed only to unlock a Bootcamp portfolio later,
not required for the High Stakes route. Lockbox still sealed.

# Cycle 6.1 (2026-07-21) — deployment-safety hardening (live/challenge NO-GO fixes)

Pre-deployment audit flagged 7 engineering defects that made live/challenge
use unsafe REGARDLESS of the strategy edge. All fixed with tests. NOTE: these
fixes make the bot SAFE TO DEPLOY; they do NOT make the STRATEGY a GO — the
edge remains NO-GO on Bootcamp under its own gates (PF≥1.25 every fold,
P(kill)<10%). The legitimate GO route is programme fit (High Stakes) and/or a
new orthogonal sleeve — not code.

Findings fixed:
1. **Risk baseline reset on restart (CRITICAL).** `RiskManager.load_or_init()`
   restores the persisted STEP-START balance from `RISK_STATE_PATH`; a restart
   after losses no longer moves the −3%/−6% kill anchor down. Daily/weekly
   baselines + counters persist too. Re-baselining a genuinely new step is a
   deliberate `main.py --new-step` action, never inferred. `save_state()` is
   called on every mutation (trade open, result, day rollover).
2. **Broker closes never reconciled (CRITICAL).** New `reconcile.py`
   (`classify_closure` pure + `reconcile_closures`) detects journalled-open
   tickets no longer open at the broker, writes the SL/TP exit, and feeds the
   result into the consecutive-loss counters. Wired into `main.process_candle`
   before the counter sync; last-seen P&L snapshot kept in `BotState`.
3. **News calendar timezone (HIGH).** `config.NEWS_SOURCE_TZ` (IANA) now drives
   FF time parsing: naive time localized to that zone then converted to UTC,
   instead of blindly assuming UTC. Verified ET→UTC conversion.
4. **No evaluation-target stop (HIGH).** `can_trade()` halts and persists a
   benign passed-step flag once equity reaches the step target AND
   `profitable_days_met()` (High Stakes ≥3 profitable days). Prevents giving
   back a qualifying result. Re-arm for the next step with `--new-step`.
5. **Official daily-loss check was dead code (HIGH).** `check_official_daily_loss()`
   is now invoked in `can_trade()` (High Stakes 5%; no-op for Bootcamp steps).
6. **Unfinished cTrader candle (MEDIUM).** New `bar_utils.drop_forming_bar()`
   excludes the still-forming trendbar in the cTrader adapter, so
   `generate_candidate` only ever sees closed bars.
7. **Failed candle not retried (MEDIUM).** `process_candle()` returns a bool;
   `main` advances `last_candle` only on success, so a transient broker/data
   failure retries the same candle instead of skipping it.

Tests: `tests/test_deployment_safety.py` (14) + `tests/test_programme_profiles.py`
(9). Full suite 249 passed; residual failures are missing-XGBoost and a
sandbox-only news-cache permission artifact — not code defects.

## Deployment gate

Engineering: demo-only until findings 1–5 are exercised on a real
restart/reconciliation cycle on the target broker (the unit/integration tests
cover the logic; a live smoke test is still required). Strategy: NO-GO on
Bootcamp; route to High Stakes (`PROGRAMME=high_stakes`) or add the Cycle-6
NY-ORB sleeve before any paid challenge.

---

## Side-study — XAUUSD copier teardown (2026-07-27)

Not a config iteration. A third-party XAUUSD-STDc copy-trading account was
analysed to decide whether any of its behaviour is reusable for Bootcamp.
Full write-up + datasets + pipeline: `research/xauusd_copier_teardown/`.

**Source material.** 16 daily MT5 statement sheets (account 29498319, USC cent
account, 29 Jun – 23 Jul 2026) + 6,281,860 Dukascopy XAUUSD ticks (1–26 Jul).
No same-broker M1/tick export and no symbol specification were supplied; every
instrument fact below is *derived* from statement arithmetic, not read from MT5.

**Reconstruction quality.** 2,917 order rows / 2,919 deal rows → 2,721 unique
deals after dropping 196 exact duplicates (Sheet15 re-states Sheet16
cumulatively). 1,356 round-trip positions matched by implied-entry-price +
FIFO, with **0.00000 USD error on 100% of matches**; daily closed P/L
reconciles **exactly on 15 of 16 sheets**. The 16th (7 Jul) differs by
−95.20 USC, traced to nine orphan closes whose opens sit in a missing 6 Jul
statement.

**Instrument facts derived.** Contract = **1 troy oz per lot**, P/L in US cents
(`profit_USC = Δprice × lots × 100`, exact on all 1,356). Leverage **≈1:500**
(22 Jul EOD: US$11,564.81 notional vs US$23.16 margin → 499.4:1). Digits 2,
point 0.01, min lot 0.02, step 0.01, commission 0, fee 0, swap net +226.66 USC.
Server clock **UTC+3**, established by offset grid search against Dukascopy
(median |price error| 0.235 USD at +3 vs 13.06 at +0) — same class of defect as
the HistData ET-vs-UTC bug in Phase 1, and worth the same paranoia everywhere.

**Strategy recovered.**
- Open 0.02 lots (92.8% of 486 baskets), no SL, no TP, market order.
- Add on ~2.10 USD/oz **adverse** movement (81.5% of 870 additions follow an
  adverse move); step is flat across levels, not expanding.
- Ladder `0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.11,0.14,0.18,0.23,0.30,0.39,0.51,0.67,0.87`
  — **428/486 baskets (88.1%) reproduce it exactly**; ascending transitions fit
  `round(prev × 1.30, 2)` at 89.8%. Fibonacci (1.618) fits only 19.2%.
- Close whole basket at ≈ **+2.3 USD/oz from weighted-average entry** (median of
  439 winners, IQR 1.78–3.01). 84% of baskets close in ≤2 timestamps.
- Max observed depth 16 levels, max 14.20 lots in one basket, max 9.9 h held.

**Entry signal: not recoverable.** 8 candidate rules tested on 442 basket starts
against all in-session M1 bars. Best precision 2.61% (fade a 5-min move >8bp)
against a 1.01% base rate. There *is* a real counter-trend tilt — longs open
after a median −4.16 bp 5-min move, shorts after +4.67 bp (Mann-Whitney
p < 0.0001), Bollinger position 0.385 vs 0.613 (p < 0.0001) — but it is far too
weak to time anything. The trigger lives on the master account.

**The number that matters.** Closed-balance drawdown over the period:
**0.00%**, 16/16 winning days, 90.3% basket win rate, PF 12.20. Tick-reconstructed
worst simultaneous floating P/L: **−7,814.97 USC = −7.67% of balance**
(−5.14% of equity, which included a 50,000 USC credit bonus). Worst single
basket **−8,465.21 USC = −8.30%**, i.e. **911× the median basket win of 9.29 USC**.
Median winning basket carried 1.11× its eventual profit in unrealised loss
before closing.

Analytically, on the fitted 16-level ladder: a **1%** adverse move → −5.8% of
balance (already past the Bootcamp limit); **2%** → −20.7%; **3%** → −35.6%.

**Statement is survivorship-filtered.** Three ledger discontinuities
(−1,373.66 / −456.94 / −1,380.28 = **−3,210.88 USC**) show losses on days whose
statements were not supplied. Real ledger change +12,738.36 vs +15,949.24
implied by the supplied trades — a **20% overstatement**. Only profitable days
were provided.

**Bootcamp compliance: 6 hard FAILs, 2 UNCLEAR, 11 PASS.** Disqualifying on
their own: external signal copying is prohibited (100% of deals are `copy #…`);
SL mandatory on every position (0/2,721, and 5 no-SL violations terminate an
account); 5% max loss per step vs −7.67% observed floating; 1:30 leverage vs
1:500 operating leverage. UNCLEAR: HFT / tick-scalping classification of 160
orders/day with sub-second 12-order bursts — needs written confirmation.

### Verdict

**NO-GO on the original bot.** Five of the nine standing NO-GO conditions are
met independently. Its 90.3% win rate *is* its risk — it is produced by refusing
to realise losses, which is the same pathology as the leakage-driven metrics
retired in the EURUSD line, only expressed through position management instead
of through the train/test split.

**REDESIGN REQUIRED** if the basket idea is pursued. Report §10–11 specifies an
independent variant: locally generated mean-reversion signal (no copier),
0.25%/basket risk, **bounded arithmetic** ladder 1.0/1.3/1.6/1.9 with max 3
additions, base lot solved backwards from a fixed `2.0 × ATR_H1` stop, shared
basket SL written onto **every** ticket, 0.75% daily / 2.0% total internal
stops, spread and ATR-percentile regime filters, 4-hour max duration, cooldown
after losses, exit-only degradation on a rejected addition. Explicitly **no**
geometric Martingale.

Gate before that redesign earns even a demo slot, in order:
1. **6–12 months of same-broker XAUUSD bid/ask ticks** + a filled-in symbol
   spec. `research_backtest.py` was run end-to-end on the 26 days available and
   produced 0/0/1 trades across a 60/20/20 split — the engine works, the data
   does not exist yet.
2. The missing 6 Jul, 13–14 Jul and 17→20 Jul statements, to attribute the
   −3,210.88 USC.
3. `P(maxDD > 5%) < 5%` on ≥20k Monte Carlo resamples — the same binary gate
   already applied to the EURUSD line.
4. Written confirmation from The5ers on the HFT / tick-scalping items.

Until all four clear, the answer stays NO-GO.

---

## GATE 0 (2026-07-27) — stop-loss counterfactual on the XAUUSD copier baskets

**Question.** Was the copier's entire return produced by the deferred-loss
mechanism? Re-simulate the 442 tick-covered baskets with a hard stop on
aggregate basket floating P/L. Entries, lot ladder and the ~+2.3 USD/oz
weighted-average-entry exit are all held identical; the only change is that a
basket dies when its floating loss reaches the stop.

**Config.** `research/xauusd_copier_teardown/gate0_stop_counterfactual.py`.
Balance anchor 101,942.72 USC (raw_summary Sheet14, 2026-06-30 EOD ledger).
Universe = 442 baskets with Dukascopy tick coverage, 2026-07-01 → 2026-07-23.
Forced-exit cost = worst-side mark (spread already in `min_float`) + 0.10
USD/oz slippage applied to `total_lots` (pessimistic: charges the full ladder
even when the stop fires before the last rung).

**Leakage verification.** No parameter is fitted here. The only input is
`min_float`, which `pipeline/tick_analysis.py` computed forward in time from
6,281,860 ticks as the running minimum of the basket's aggregate floating P/L
counting only positions already opened at each second (`live = tsec >= opens`).
A basket stops iff `min_float <= -stop`, because the path up to the first
crossing is identical in the counterfactual and the stop merely truncates it.
The four stop levels are fixed fractions of a balance known at period start,
not quantiles of the outcome distribution.

| Config | Stopped | Net USC | Net % | WR % | PF | Worst loss | Max DD % |
|---|---|---|---|---|---|---|---|
| As traded (no stop) | 0 | +13,911.32 | +13.646 | 90.95 | 12.088 | −882.44 | 0.866 |
| Stop 0.5% (510 USC) | 19 | −4,233.89 | −4.153 | 87.10 | 0.597 | −651.71 | 4.153 |
| Stop 1.0% (1,019 USC) | 12 | −4,189.43 | −4.110 | 88.69 | 0.678 | −1,161.43 | 5.453 |
| Stop 2.0% (2,039 USC) | 6 | −3,284.01 | −3.221 | 89.82 | 0.745 | −2,180.85 | 5.835 |
| Stop 3.0% (3,058 USC) | 3 | +1,013.70 | +0.994 | 90.50 | 1.104 | −3,200.28 | 3.139 |

Idealised zero-slippage fills change nothing material: −3,769 / −3,798 / −2,990
/ +1,229 USC at the four levels.

**Verification (rule 6 applied to PF 12.088 and to the size of the swing).**
- 0.5% counterfactual reconciles exactly: 13,911.32 − 18,145.21 = −4,233.89.
- The swing comes from 19 of 442 baskets (4.3%). 17 of those 19 were *winners*
  worth 8,882.07 USC as traded.
- The deepest-underwater baskets are the *largest winners*. S241 (min_float
  −8,465.21 = −8.30% of balance) returned +4,021.16 USC — 28.9% of the entire
  period's profit from one basket that was 8.3% underwater. S042 (−2.15%) →
  +955.83. Top 10 baskets = 62.8% of net. 217 of 402 winners carried a peak
  floating loss larger than their eventual profit.
- 0 of 442 baskets were never underwater.
- 78 of 441 adjacent basket pairs overlap in time, so the closed-balance max DD
  above **understates** the true equity drawdown under a stop regime.

**Survivorship adjustment.** Adding back the −3,210.88 USC of ledger
discontinuities on the unsupplied days (§2.2 of the teardown report) moves the
four levels to −7.30% / −7.26% / −6.37% / **−2.16%**. The 3% level's profit
does not survive the known 20% overstatement.

### Verdict: GATE 0 FAILED — stop.

- **Pass criterion ("net positive at a stop of 2% or tighter") is not met**:
  2% → −3.22%, 1% → −4.11%, 0.5% → −4.15%. Profit factor is below 1.0 at every
  one of these levels.
- The only positive cell, 3%, is not a rescue: it is +0.99% on 17 trading days
  in a single regime, PF 1.104 (below the 1.2 threshold Gate 3 would have
  applied), it goes to −2.16% once the known survivorship gap is restored, and
  a 3% single-basket loss consumes 60% of the entire Bootcamp 5% step budget in
  one trade. It is not a deployable stop level; it is the level at which the
  stop stops binding.
- Max DD flips from 0.866% to 5.45–5.84% at the 1–2% stops — i.e. imposing a
  stop **breaches the 5% Bootcamp limit** rather than protecting against it,
  because the stop converts hidden floating loss into realised loss without
  changing the underlying price paths.

**Reading.** The copier's return was the deferred-loss mechanism, in full. The
90.3% win rate and PF 12.09 were not an edge being harvested; they were losses
being postponed, and the biggest wins were the baskets that postponed the most.
Once losses are realised on any schedule tight enough to be compliant, the
expectancy is negative. No parameter was tuned to rescue this and none will:
the entry has no demonstrated directional edge (best precision 2.61% vs a 1.01%
base rate, §5 of the teardown), so the ladder is redistributing a negative
expectancy, not creating a positive one.

**Gates 1–4 not run.** Investigation halted at Gate 0 per protocol, pending
decision.

---

## STEP 1 (2026-07-27) — feasibility envelope for ablation (a), post-Gate-0

**Question.** Gate 0 killed the XAUUSD basket line. Ablation (a) — regime-only
entry, first valid in-session bar per day — is the only positive result in this
repo that survived scrutiny. Before building anything on it: does ANY
risk-per-trade setting simultaneously give a usable probability of reaching the
programme target AND hold `P(maxDD > 5%) < 5%`?

**Config.** `research/feasibility_envelope.py`. R series from
`research_ablation.py` variant (a), unchanged. 20,000 block-bootstrap paths,
block=10, seed=42. maxDD measured on unabsorbed paths over a 111-trade (1-year)
horizon. Lockbox 2025-03-20 NOT touched.

**Leakage verification.** The R series comes from the existing bar-by-bar
walk-forward engine: every indicator is trailing, SL/TP come from the frozen
production clamp evaluated at the entry bar, exits are walked forward tick-order
(stop before target). The risk sweep is a fixed a-priori grid, not fitted to
outcomes. No quantity in this study is estimated from data after the trade it
is applied to.

### 1a. Reproduction — n matches exactly, R does not

| | logged (Phase 2c) | now |
|---|---|---|
| n | 1128 | **1128** |
| PF | 1.118 | 1.185 |
| E[R] | +0.064 | +0.0965 |

Entry count reproduces to the trade, so the entry logic is unchanged; the R
values moved because the exit/cost layer changed after Phase 2 (Cycle 4
re-verified The5ers costs). **The current numbers are more favourable than the
logged ones.** Recorded rather than adopted silently; all figures below use the
current verified cost layer, and the direction of the discrepancy is noted.

Data now extends to 2007 (474,790 M15 bars), not 2015. Design window
(2015-01 → 2025-03-19) kept as the protocol window; 2007-2025 reported as
sensitivity only.

### 1b. Is the edge real? (the Gate-2 question, asked of variant (a))

Design window: **E[R] = +0.0965, 95% CI [+0.0197, +0.1744], t = 2.44,
P(E[R] ≤ 0) = 0.0066.** Block bootstrap agrees ([+0.0185, +0.1774]).
Full 2007-2025: E[R] = +0.1318, CI [+0.0766, +0.1874], P(E[R] ≤ 0) = 0.000.

**The edge clears a 95% test.** It is the first thing in this repo that has.

### 1c. But it is not stationary

| Subperiod | n | PF | E[R] | 95% CI | P(E[R] ≤ 0) |
|---|---|---|---|---|---|
| 2007-2011 | 663 | 1.293 | +0.1501 | [+0.048, +0.253] | 0.002 |
| 2012-2014 | 436 | 1.384 | +0.1891 | [+0.062, +0.316] | 0.002 |
| 2015-2017 | 402 | 1.404 | +0.1957 | [+0.065, +0.329] | 0.001 |
| **2018-2020** | 305 | 1.013 | **+0.0072** | **[−0.136, +0.155]** | **0.463** |
| **2021-2025** | 427 | 1.139 | **+0.0740** | **[−0.049, +0.200]** | **0.125** |

First half (2015-2019) +0.1459R; second half (2020-2025) +0.0426R. Every
subperiod from 2018 onward has a confidence interval containing zero.

Counter-evidence, stated for balance: the OLS trend of R on trade index has
**t = −1.32 — the decay is NOT statistically significant**, 92% of rolling
250-trade windows are positive, and 8 of 11 years are positive. So "the edge
is decaying" is not established; what IS established is that the post-2018
data alone cannot distinguish the edge from zero. Both readings are true.

### 1d. The envelope — Bootcamp (+6% target, −5% static, −3% kill)

| risk/trade | P(pass step 1) | P(kill) | P(breach) | median trades | ~months | **P(maxDD>5%)** | P(all 3 steps) |
|---|---|---|---|---|---|---|---|
| 0.15% | 0.901 | 0.099 | 0.0000 | 316 | 34.2 | **0.0022** ✅ | — |
| 0.18% | 0.857 | 0.143 | 0.0000 | 244 | 26.4 | **0.0088** ✅ | — |
| 0.20% | 0.829 | 0.171 | 0.0000 | 211 | 22.8 | **0.0199** ✅ | 0.569 |
| 0.22% | 0.801 | 0.199 | 0.0000 | 184 | 19.9 | **0.0357** ✅ | — |
| **0.23%** | 0.788 | 0.212 | 0.0000 | 171 | 18.5 | **0.0449** ✅ | — |
| 0.25% | 0.765 | 0.235 | 0.0000 | 150 | 16.2 | **0.0691** ❌ | 0.447 |
| **0.30% (current config)** | 0.712 | 0.288 | 0.0000 | 113 | 12.2 | **0.1532** ❌ | 0.361 |
| 0.50% | 0.575 | 0.425 | 0.0000 | 48 | 5.2 | **0.5462** ❌ | 0.190 |
| 1.00% | 0.463 | 0.537 | 0.0000 | 15 | 1.6 | **0.9588** ❌ | 0.099 |

`P(reach +6% before ever touching −X)`, Bootcamp step 1, kill switch disabled:

| risk | X=1% | X=2% | X=3% | X=4% | X=5% |
|---|---|---|---|---|---|
| 0.20% | 0.465 | 0.699 | 0.829 | 0.904 | 0.945 |
| 0.25% | 0.410 | 0.626 | 0.765 | 0.851 | 0.906 |
| 0.30% | 0.365 | 0.571 | 0.712 | 0.803 | 0.867 |

`P(breach) = 0.0000` at every risk level is mechanically correct, not a bug:
worst single R = −1.296, so at ≤0.30% risk one trade moves ≤0.39% and cannot
gap the −3% kill through to −5%. The maxDD column is the honest risk measure
because it does not assume the kill switch fires.

### 1e. High Stakes (10%+5% targets, −10% static, −6% kill)

| risk | P(pass step 1) | median trades | ~months | P(all steps) |
|---|---|---|---|---|
| 0.20% | 0.968 | 423 | 45.8 | 0.938 |
| 0.25% | 0.938 | 321 | 34.7 | 0.884 |
| 0.30% | 0.903 | 251 | 27.2 | 0.825 |

Confirms the Cycle-6 finding directionally: the symmetric budget suits a thin
edge far better. It buys completion probability with time, not with less risk —
the maxDD column is identical (it is a property of the return stream).

### Verdict: STEP 1 PASSED, narrowly — with a hard constraint attached.

A feasibility window exists, and it is **risk ≤ 0.23% per trade**. That is the
entire window. Above it the strategy fails its own `P(maxDD>5%) < 5%` gate.

**Live config defect found.** `config.PROGRAMMES["bootcamp"]["risk_per_trade_pct"]
= 0.003` (0.30%) yields **P(maxDD > 5%) = 15.3% — three times over the hard
gate**. The current default is not compliant with the repo's own standard and
must come down to ≤0.0023 before any challenge. High Stakes at 0.004 (0.40%)
is worse still at 35.2%.

**What the window actually buys.** At 0.20% risk: 82.9% chance of passing
Bootcamp step 1, median **211 trades ≈ 22.8 months**. All three steps: 56.9%,
on the order of five years. This edge is real and it is compliant, but it is
too thin to pass a 3-step +6% programme on any timescale a person would call
a plan.

**Recommendation before Step 2.** Do not start the faithful reimplementation
yet. The binding constraint is not implementation quality, it is that
+0.0965R × 111 trades/year cannot clear +6% quickly enough at a risk level that
survives the drawdown gate. The productive next question is trade FREQUENCY,
not signal quality: (a) carries one entry per day per instrument by
construction, and the multi-instrument pooling work (Lever 1) is the only route
that raises trades/year without raising risk/trade. Awaiting decision.

---

## GATE A (2026-07-27) — does ablation (a) replicate per instrument?

**Question.** Lever 1 (pooled multi-instrument) only helps if the added
instruments carry edge, not just frequency. Run ablation (a) unchanged on each
of six instruments, design window 2015-01-01 → 2025-03-19 (lockbox 2025-03-20
untouched). Kill if fewer than 3 have a bootstrap CI on E[R] with a lower bound
above 0.

**Config.** `research/gate_a_per_instrument.py`. R series from
`research_ablation.run_ablation` variant (a), unchanged entry logic. The
ablation was made symbol-aware (surgical): `_prepared_engine`/`run_ablation`
now thread `symbol=` to `RulesBacktestEngine`, `_simulate` uses the engine's
own asset-aware `eng._sl_tp`, and variant (a)/(c) entries use `eng._entry_cost`.
For EURUSD (bp_mode off, default mults) these are byte-identical to the prior
FX path — **guardrail: EURUSD reproduces n=1128, PF 1.185, E[R] +0.0965 to the
trade**, confirming the edits are neutral. For metals/index the change applies
the correct price-relative clamp/cost instead of the EURUSD pip clamp.

**Leakage verification.** R comes from the bar-by-bar walk-forward engine:
every indicator/threshold is trailing, SL/TP from the frozen entry-bar clamp,
exits walked forward tick-order (SL before TP). Design window ends before the
2025-03-20 lockbox, which is not read. Bootstrap resamples the realised R
vector only — no future information enters.

| Symbol | n | tr/yr | PF | E[R] | iid 95% CI | block 95% CI | P(E[R]≤0) | 2018+ E[R] [CI] | edge? |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 1128 | 111 | 1.185 | +0.0965 | [+0.0195,+0.1754] | [+0.0172,+0.1759] | 0.0073 | +0.0461 [−0.050,+0.142] | ✅ |
| GBPUSD | 1547 | 152 | 1.218 | +0.1132 | [+0.0480,+0.1797] | [+0.0450,+0.1830] | 0.0004 | +0.0919 [+0.010,+0.173] | ✅ |
| AUDUSD | 1104 | 109 | 1.041 | +0.0224 | [−0.0521,+0.0989] | [−0.0534,+0.0978] | 0.2784 | −0.0427 [−0.136,+0.053] | ❌ |
| USDJPY | 1138 | 112 | 1.043 | +0.0241 | [−0.0531,+0.1022] | [−0.0491,+0.0985] | 0.2688 | −0.0167 [−0.113,+0.079] | ❌ |
| XAUUSD | 1500 | 148 | 0.968 | −0.0182 | [−0.0835,+0.0480] | [−0.0866,+0.0503] | 0.7046 | −0.0334 [−0.112,+0.046] | ❌ |
| GRXEUR |  676 |  67 | 1.184 | +0.0987 | [−0.0019,+0.2024] | [+0.0002,+0.1955] | 0.0277 | +0.1076 [−0.019,+0.236] | ❌ (iid straddles 0) |

Only **EURUSD and GBPUSD** clear both bootstrap CI lower bounds above 0.
GRXEUR is the near-miss: block CI barely clears (+0.0002) but the iid CI
straddles zero (−0.0019) and it carries the fewest trades (67/yr). AUDUSD and
USDJPY are indistinguishable from a coin flip (PF ~1.04, P(E[R]≤0) ~0.27).
XAUUSD is negative on this signal (PF 0.968).

**Pairwise daily-aggregate-R correlation (design window), reported for balance —
this is the Gate-B question, answered early because it changes how the kill
reads:**

```
        EURUSD  GBPUSD  AUDUSD  USDJPY  XAUUSD  GRXEUR
EURUSD    1.00    0.18    0.13    0.04    0.15    0.09
GBPUSD    0.18    1.00    0.12    0.02    0.07    0.02
```

The survivors are **not** correlated (EUR–GBP r = 0.175). So the kill is NOT
"one trade in six hats" — realized outcomes are near-independent across the
board (max pairwise 0.18). The kill is narrower and harder: **only two of six
instruments carry any edge at all.** Pooling the other four raises trades/year
by importing two coin flips (AUD, JPY) and one losing stream (XAU) — precisely
the failure the gate was built to catch: "pooling uncorrelated noise raises
frequency without raising expectancy."

### Verdict: GATE A FAILED — stop.

- **Kill criterion met:** 2 of 6 instruments clear the CI test, below the
  required 3. Even counting GRXEUR generously (block-only) reaches 3 by one
  instrument that (i) fails the iid test, (ii) adds the least frequency, and
  (iii) is a different asset class (index) whose transfer is least established.
- The two real edges, EUR + GBP, are low-correlation (r=0.18) and would give
  genuine breadth — but that is two instruments, ~263 combined trades/year, and
  the protocol threshold is ≥3. Reporting it for the decision, not overriding
  the gate.
- **Reading.** The binding constraint identified in Step 1 (frequency, not
  signal quality) is not relieved by this instrument set. The signal is
  specific to EUR/European-session USD majors; it does not transfer to AUD, JPY,
  gold or the DAX. Naive pooling would degrade expectancy, not just fail to
  raise it.

**Gates B–E not run.** Halted at Gate A per protocol, pending decision. A
EUR+GBP two-instrument pool is the only configuration the data supports raising
if the ≥3 threshold is relaxed — that is the user's call, not a workaround.

---

## GATE 0 (2026-07-28) — reproduce & reconcile EURUSD H1 regime; LEAKAGE FOUND

**Scope.** Phase-1 signal-validation: reproduce the frozen H1 regime signal
(ablation variant (a), first eligible in-session bar/day), reconcile the
+0.064R↔+0.0965R history, and check look-ahead before any cross-market test.
Scripts: `research/gate0_reconcile.py`, `research/gate0_leakage_probe.py`,
`tests/test_regime_alignment_leakage.py`.

### 1. Reproduction (current cost, CURRENT alignment) — both windows exact
- **Window A 2015-01→2025-03:** n=**1128** (exact), E[R] **+0.0965**, med −0.149,
  sd 1.329, WR 31.5%, PF 1.185, maxDD 23.8R, longest losing streak 18,
  111/yr, **8/11 years positive**. iid95% [+0.020,+0.175], block95% [+0.018,+0.176].
  Long n=558 +0.138R (PF 1.273); short n=570 +0.056R (PF 1.104).
- **Window B 2007-01→2025-03:** n=**2233** (exact), E[R] **+0.1318**, WR 32.7%,
  PF 1.257, 123/yr, **15/19 years positive**. iid95% [+0.076,+0.187].
- Both reproduce their targets to the trade. Windows reported separately.

### 2. +0.064R vs +0.0965R — fully reconciled to the Cycle-4 cost change
Held the 1,128 entries and directions fixed (n=1128 in every cell — the cost
model does not touch the entry set, because the SL clamp is on ATR distance,
not entry price). Cost layer is the ONLY difference (exit params byte-identical
between commit af4e876 and HEAD).

| spread / commission | E[R] | PF |
|---|---|---|
| 0.6 pip / $7 RT (pre-Cycle-4) | **+0.0644** | 1.118 |
| 0.4 pip / $4 RT (Cycle-4 verified, current) | **+0.0965** | 1.185 |
| 0.6 pip / $4 RT | +0.0912 | 1.173 |
| 0.4 pip / $7 RT | +0.0697 | 1.129 |

Additive decomposition of the +0.0321R gap: **commission $7→$4 = +0.0268R
(83.5%)**, spread 0.6→0.4 pip = +0.0053R (16.5%); the two effects sum exactly
(no interaction). The historical discrepancy is 100% cost, 0% signal, exactly
as pre-diagnosed. Not a reproduction failure.

### 3. LEAKAGE — the H1→M15 regime alignment reads a not-yet-closed bar
The H1 CSVs are **left-labelled** (bar `o` spans [o, o+1h), closes at o+1h;
verified: H1 open=first M15 open, H1 close=last M15 close in the window).
`strategy.build_signal_frame` aligns regime with `merge_asof(direction=
"backward")` on the H1 **label**, so an M15 bar at :00/:15/:30 is matched to the
H1 bar **containing** it — still forming, closing up to 45 min in the future.
Its regime uses `close ≷ ema_fast` on that future close.

Concrete instance (verified): M15 2015-01-05 09:15 is matched to H1 label 09:00,
which covers [09:00,10:00) and closes at 10:00 — after the M15 bar. The last H1
bar actually closed by 09:15 is the 08:00 bar. Production uses the forming bar.

**Honest alignment** = regime of the most recent H1 bar CLOSED by the entry
bar's timestamp (align on close time = label+1h). Variant (a), Window A:

| pair | leaky n | leaky E[R] | leaky PF | honest n | honest E[R] | honest PF | honest iid95% |
|---|---|---|---|---|---|---|---|
| EURUSD | 1128 | +0.0965 | 1.185 | 1100 | **−0.1223** | 0.805 | [−0.197,−0.046] |
| GBPUSD | 1547 | +0.1132 | 1.218 | 1505 | **−0.1164** | 0.812 | [−0.180,−0.051] |
| AUDUSD | 1104 | +0.0224 | 1.041 | 1071 | **−0.1141** | 0.812 | [−0.190,−0.037] |
| USDJPY | 1138 | +0.0241 | 1.043 | 1107 | **−0.1291** | 0.791 | [−0.203,−0.054] |

**Localisation of the edge (EURUSD, variant (a) entries split by bar):**
- common-bar entries (present under both alignments), n=634: E[R] **−0.1216**
- leak-only entries (intra-hour, created by the forming-bar look-ahead), n=499:
  E[R] **+0.3737**
- honest-only entries, n=467: E[R] −0.1232

The entire positive expectancy is carried by the 499 intra-hour entries whose
regime label already "knows" the direction its own H1 bar will close in. Remove
the look-ahead and every pair is significantly negative, CI entirely below 0.
Spot-check confirmed the alignment logic: 100/100 top-of-hour entry bars agree
leaky==honest (as required — those legitimately use the just-closed bar).

### Verdict: GATE 0 FAILED — leakage found. STOP.
Per the Gate-0 fail criteria ("Fail if leakage is found"): the frozen H1 regime
signal's positive expectancy is a look-ahead artifact of the `merge_asof`
regime alignment, not an edge. It is **structural, not EURUSD-specific** — the
same mechanism makes all four FX pairs significantly negative once corrected.
This retroactively explains the positive design-window numbers across the prior
cycles (all used `build_signal_frame`) and is consistent with the flat/negative
lockbox (Phase 4: PF 0.894), which was itself leaky and therefore *over*stated.

**Gate 1, binding-cap diagnostic, and the Phase-1 decision options are moot**
(there is no honest edge to replicate). Documented, not rescued: no parameter
tuned, no pair removed, no exit altered.

**Not applied, pending decision:** the fix is a production change to
`strategy.build_signal_frame` (align regime on H1 close time / last closed bar),
which flips the strategy from "candidate" to negative and changes live
behaviour. `tests/test_regime_alignment_leakage.py` encodes the no-look-ahead
invariant (xfail(strict) now; flips to pass when the alignment is fixed).

### Decision memo
`STOP — the H1 regime signal has no honest edge on EURUSD or any of the four FX
pairs; the reproduced +0.0965R/+0.1318R are look-ahead artifacts.` (This does
not match the pre-written STOP labels because the failure mode is leakage, which
they did not anticipate.) Awaiting decision on whether to (a) apply the
alignment fix and re-baseline the whole line as a leak-free negative result, or
(b) retire the H1 regime line.
