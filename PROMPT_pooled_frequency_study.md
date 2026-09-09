# Prompt — pooled multi-instrument frequency study (Lever 1)

Paste into Claude Code from the root of `forex_bot` on branch `the5ers-bootcamp`.

---

You are a quantitative researcher working in this repo. Read `research_log.md`
from the end backwards — at minimum the two most recent entries, **GATE 0
(2026-07-27)** and **STEP 1 (2026-07-27)** — before doing anything. Then read
`research_ablation.py`, `research_pooling.py`, `research/feasibility_envelope.py`,
`monte_carlo_dd.py` and the `PROGRAMMES` block in `config.py`.

## Where the investigation stands

Two lines are dead and are not to be revisited:

- **XAUUSD copier basket** — killed at Gate 0. With a hard basket stop the net
  P/L is −4.15% / −4.11% / −3.22% at 0.5% / 1% / 2% stops, PF < 1.0 at every
  level. The 90.3% win rate was deferred loss, not edge.
- **EURUSD full trigger** (pullback + RSI recross) — PF 0.998 on n=1278. The
  M15 confirmation subtracts from the H1 regime it sits on.

One thing is alive. Ablation variant **(a)**, regime-only entry, first valid
in-session bar per day:

| | |
|---|---|
| Design window 2015-01 → 2025-03-19 | n=1128, PF 1.185, **E[R] +0.0965** |
| 95% CI on E[R] | **[+0.0197, +0.1744]**, t=2.44, P(E[R]≤0)=0.0066 |
| Full 2007 → 2025-03-19 | n=2233, PF 1.257, E[R] +0.1318 |
| Trade frequency | **111/year on EURUSD alone** |
| Feasibility window vs P(maxDD>5%)<5% | **risk ≤ 0.23%/trade** |
| At 0.20% risk | P(pass Bootcamp step 1) 0.829, median **211 trades ≈ 22.8 months** |

Known non-stationarity, to be carried forward and not swept under: every
subperiod from 2018 onward has a CI containing zero (2018-2020 E[R] +0.0072,
CI [−0.136, +0.155]; 2021-2025 +0.0740, CI [−0.049, +0.200]). The decay trend
itself is **not** significant (t = −1.32) and 92% of rolling 250-trade windows
are positive. Both facts are true. Report both, always.

## Objective

Step 1 established that the binding constraint is **trades per year, not signal
quality**. +0.0965R × 111 trades/year cannot reach +6% quickly enough at a risk
level that survives the drawdown gate.

Determine whether running ablation (a) across multiple instruments raises
frequency enough to make the Bootcamp reachable on a usable timescale, **without
raising portfolio drawdown past the hard gate**. If it does not, prove that
cleanly and stop.

Candidate instruments (all have `data/<SYM>_M15_real.csv` + `_H1_real.csv`):
`EURUSD` (474,791 M15 bars), `GBPUSD` (479,453), `AUDUSD` (280,638),
`USDJPY` (280,679), `XAUUSD` (266,576), `GRXEUR` (146,577).

## Standing rules

1. **Research only.** No live orders, no `MetaTrader5` import in any research
   path, no broker sockets. Reuse the guard pattern at the top of
   `research/xauusd_copier_teardown/research_backtest.py` and
   `research/feasibility_envelope.py`.
2. **No leakage.** Every indicator, percentile, threshold and scaler comes from
   a trailing window only. Before reporting any result, state in one line how
   you verified no future information entered it. A result you cannot explain
   that way does not get reported.
3. **Reuse the existing stack.** `research_ablation.py` (variant (a) is already
   implemented — do not rewrite it), `research_pooling.py`, `risk_manager.py`,
   `monte_carlo_dd.py`, `validation.py`, `symbol_specs.py`. No parallel risk
   layer.
4. **Negative results are the deliverable.** If it does not work, say so with
   the number that shows it. Do not soften it, do not go looking for a variant
   that looks better.
5. **Append every gate outcome to `research_log.md`** in the existing style:
   config, numbers, verdict. Commit after each gate with a descriptive message.
   Path-scope your `git add` — there are pre-existing uncommitted changes in
   `config.py`, `main.py`, `backtest.py`, `README.md` that are not yours.
6. **If a number looks extraordinary** (Sharpe > 3, WR > 80%, PF > 5, drawdown
   near zero, or a metric that improves right after you change something),
   treat it as a bug until proven otherwise and go find the bug before
   reporting it. That heuristic has now been right three times in this repo.
7. **Do not tune to rescue a failed gate.** Report the failure, log it, stop for
   my decision.

## GATE A — does (a) replicate per instrument?

Run ablation (a) unchanged on each of the six instruments, design window
2015-01-01 → 2025-03-19 (lockbox starts 2025-03-20 and is not touched).

Report per instrument: n, trades/year, PF, E[R], 95% bootstrap CI on E[R]
(iid **and** block, block=10), and the 2018+ subperiod E[R] with its CI.

**Kill criterion:** fewer than 3 instruments have a bootstrap CI on E[R] whose
lower bound is above 0. Pooling uncorrelated *noise* raises frequency without
raising expectancy — it just converts a slow coin flip into a fast one.

## GATE B — how much of that frequency is independent?

This is the gate that decides the study. Naive breadth is a lie if the
instruments trade the same thing at the same time.

- Build the pooled trade stream in **calendar order**, preserving concurrency.
  Do **not** concatenate per-symbol streams — that destroys the clustering that
  produces drawdown, and it is the exact analogue of the leakage bug that killed
  the EURUSD line.
- Note that `research_pooling.py` currently enforces **one open trade globally**.
  That constraint caps frequency at roughly the single-instrument rate and so
  defeats the purpose. Run it both ways — one-global-trade (the existing
  constraint) and concurrent-positions-allowed — and report both.
- Report: pairwise correlation of daily aggregate R across instruments; the
  distribution of simultaneously-open positions; peak concurrent risk as a
  multiple of risk-per-trade; and **effective breadth** = (Σw)² / Σ(w'Σw),
  i.e. the number of genuinely independent bets.

**Kill criterion:** effective breadth < 2.0. Below that, pooling is one trade
wearing six hats and the drawdown gate will bind at the same risk level as
EURUSD alone.

## GATE C — pooled feasibility envelope

Rerun `research/feasibility_envelope.py` logic against the pooled stream.
Sweep risk-per-trade 0.05% → 0.30%. For each: P(pass step 1), P(kill),
P(breach), median trades and **median months** to pass, and P(maxDD > 5%)
measured on unabsorbed paths over a one-year horizon.

Bootstrap must resample **calendar blocks** (e.g. 2-week blocks of the pooled
stream), not individual trades, so concurrency and clustering survive the
resample.

**Hard gate:** `P(maxDD > 5%) < 5%`.
**Kill criterion:** no risk level clears the hard gate with a median
time-to-pass on step 1 of **≤ 9 months**. At 22.8 months the single-instrument
version is already unusable; if pooling cannot get inside 9, it has not solved
the problem it was brought in to solve.

## GATE D — lockbox, touched once

Only if Gate C passes. Run the single winning configuration once on
2025-03-20 → 2026-03-20. Report PF, E[R], maxDD, and the fold table.

**Kill criterion:** PF < 1.2 out-of-sample, or the result is a spike rather
than a plateau across the risk/instrument-set surface.

## GATE E — survivability and programme fit

≥20,000 Monte Carlo paths: calendar-block resampling, slippage resampling,
rejection-rate resampling. Report `P(reach target before breaching X)` for
X = 1, 2, 3, 4, 5%.

**Hard gate:** `P(maxDD > 5%) < 5%`. Binary. NO-GO above it regardless of
return.

Then recommend a programme on the Monte Carlo output — Bootcamp (3× +6%, −5%
static, −3% kill) vs High Stakes (10%+5%, −10% static, −6% kill) — with breach
probabilities and median months-to-completion for each. Do not recommend a
programme switch purely to escape a constraint that is protecting me.

## Housekeeping, independent of the gates

1. **Live config defect, fix first.** `config.PROGRAMMES["bootcamp"]
   ["risk_per_trade_pct"] = 0.003` gives P(maxDD > 5%) = **15.3%**, three times
   over the repo's own hard gate. `["high_stakes"] = 0.004` gives 35.2%. Bring
   both inside the gate, update the comments that justify the old values, and
   add a test that fails if any programme profile's risk setting implies
   P(maxDD>5%) ≥ 5% on the current R distribution.
2. **Git locks.** `.git/HEAD.lock` and `.git/index.lock` are stale and blocked
   the Step 1 commit. `rm -f .git/HEAD.lock .git/index.lock`, then commit the
   Step 1 artifacts already on disk: `research_log.md`,
   `research/feasibility_envelope.py`, `research/feasibility_envelope.csv`,
   `research/xauusd_copier_teardown/gate0_stop_counterfactual.py`.
3. **Secret in git history.** `.env` with a real MT5 account number is in
   history — added in `c4ec4c1`, removed in `8f4e608`, still recoverable across
   all 27 commits. Purge with `git filter-repo` (preferred) or BFG, force-push,
   and tell me to rotate the credentials on the assumption they are already
   exposed. Do not do the force-push without confirming with me first.

## Working style

Start with Gate A and report before moving on. At each gate give me the number
and the verdict in a few lines — I do not need the narration. Do not proceed
past a failed gate. If you hit something that changes the shape of the
investigation, stop and say so rather than working around it.
