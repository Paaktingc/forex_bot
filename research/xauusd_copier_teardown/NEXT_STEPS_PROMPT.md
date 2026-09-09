# Prompt — XAUUSD basket strategy: edge validation and build

*Paste into Claude Code from the root of `forex_bot` on branch `the5ers-bootcamp`.*

---

You are a quantitative researcher working in this repo. Read
`research/xauusd_copier_teardown/README.md` and
`research/xauusd_copier_teardown/XAUUSD_Bot_Reverse_Engineering_Report.md`
first, then `research_log.md` (the whole file — the EURUSD line already died of
data leakage and a directionless primary model, and I need you to not repeat it).

**Objective:** determine whether a risk-capped XAUUSD basket strategy has a real,
out-of-sample edge, and if it does, build it to deployment standard. If it does
not, prove that cleanly and stop.

This is a **gated** investigation. Each gate has an explicit kill criterion. Do
not proceed past a failed gate. Do not tune parameters to rescue a failed gate —
report the failure, update `research_log.md`, and stop for my decision.

---

## Standing rules

1. **Research only.** No live orders, no MetaTrader5 import in any research path,
   no broker sockets. `research/xauusd_copier_teardown/research_backtest.py` has
   the guard pattern — reuse it.
2. **No leakage.** Any indicator, percentile, threshold or scaler must be
   computed from a trailing window only. Before reporting any result, state in
   one line how you verified no future information entered it. A result you
   cannot explain this way does not get reported.
3. **Reuse the existing risk layer.** `risk_manager.py` (step-start anchor,
   daily/total kill switch, persisted state), `monte_carlo_dd.py`,
   `validation.py`, `symbol_specs.py`. Do not build a parallel risk stack.
4. **Negative results are the deliverable.** If something does not work, say so
   plainly with the number that shows it. Do not soften it and do not search for
   a variant that looks better.
5. **Append every gate outcome to `research_log.md`** in the existing style:
   config, numbers, verdict. Commit after each gate with a descriptive message.
6. If a number looks extraordinary (Sharpe > 3, win rate > 80%, PF > 5, drawdown
   near zero), treat it as a **bug until proven otherwise** and go find the bug
   before reporting it. That heuristic has already been right twice in this repo.

---

## GATE 0 — Stop-loss counterfactual (do this first, it is cheap and decisive)

Using `research/xauusd_copier_teardown/data/trades_reconstructed.csv`,
`baskets_with_mae.csv` and the Dukascopy tick file, re-simulate all 442
tick-covered baskets **with a hard basket stop-loss** that the original bot did
not have. Stop levels to test: 0.5%, 1%, 2%, 3% of the 101,942 USC balance,
applied to the basket's aggregate floating P/L.

Keep everything else identical: same entries, same lot ladder, same exit at
weighted-average entry + ~2.3 USD/oz. The only change is that a basket now dies
when its floating loss hits the stop.

Report, per stop level: total net P/L, win rate, profit factor, number of
baskets stopped out, worst realised loss, and the resulting max drawdown.

**Kill criterion:** if total net P/L is negative at every stop level ≤ 3%, the
strategy's entire return was the deferred-loss mechanism. There is nothing to
salvage. Stop and report.

**Pass:** net P/L stays positive at a stop of 2% or tighter. Continue.

---

## GATE 1 — Data

Acquire and validate 12 months of XAUUSD data, 2025-07 → 2026-07:

- Same-broker `XAUUSD-STDc` M1 + ticks if I can export them (ask me — I may need
  to do this manually in MT5)
- Dukascopy XAUUSD M1 + tick as the working feed
  (`research/xauusd_copier_teardown/` has the download script; widen the range)

Run it through `prepare_data.py` / `resampler.py`. Validate exactly as
`tick_analysis.py` does: monotonic timestamps, duplicates, null bid/ask, crossed
spreads, gap inventory, spread distribution. **Explicitly confirm the timezone**
against a known reference — the HistData ET-vs-UTC bug and the UTC+3 broker
offset in the teardown were both silent killers.

**Kill criterion:** < 9 months of clean data. Stop and tell me what is missing.

---

## GATE 2 — Does the signal have directional edge, independent of basket structure?

This is the question the EURUSD line failed. Test it **before** building any
basket logic, because basket management cannot manufacture edge.

Take the proposed entry (Bollinger M15 rejection + RSI extreme + ATR regime
filter + EMA200 distance filter, spec in report §10.3) and evaluate it as a
**plain single-entry trade** with a fixed stop and target — no grid, no
averaging, no basket.

Required comparisons:
- vs a random-entry baseline matched on count, session and holding time
- vs a shuffled-label baseline (destroy the signal, keep the sampling)

Report hit rate, average R, and the bootstrap confidence interval on average R.

**Kill criterion:** average R is not distinguishable from the random baseline at
95% confidence. That means no directional edge and the basket structure will
only redistribute the same negative expectancy. Stop and report.

---

## GATE 3 — Full strategy, walk-forward

Implement the strategy in report §10–11 (bounded arithmetic ladder 1.0/1.3/1.6/1.9,
max 3 additions, base lot solved backwards from a 2.0 × ATR_H1 stop, shared
basket SL on every ticket, 0.25% risk/basket, 0.75% daily and 2.0% total internal
stops, spread + ATR-percentile filters, 4h max duration, cooldown after losses,
exit-only degradation on a rejected addition).

Backtest on **ticks**, not candle closes, with variable spread, commission, swap,
slippage, execution delay, order rejection and intratick sequencing (stop checked
before target).

Chronological split: 60% discovery / 20% selection / 20% out-of-sample, plus
walk-forward (rolling 3-month train, 1-month test). Touch the out-of-sample
window **once**.

Then report parameter-stability surfaces. I want to see a **plateau**, not a
spike — a result that only exists at one parameter setting is noise.

**Kill criterion:** out-of-sample profit factor < 1.2, or performance is a spike
rather than a plateau.

---

## GATE 4 — Survivability

Monte Carlo, ≥ 20,000 paths: trade-order resampling, slippage resampling,
rejection-rate resampling.

Report `P(reach target before breaching X)` for X = 1%, 2%, 3%, 4%, 5%.

**Hard gate:** `P(maxDD > 5%) < 5%`. This is binary, same as the EURUSD line.
Anything above it is a NO-GO regardless of how good the return looks.

---

## Also decide, at Gate 4

Compare programme fit on the Monte Carlo output: **Bootcamp** (6% target, 5% max
loss per step, 1:30, mandatory SL) vs **Hyper Growth** (10% target, 6% stop-out,
3% daily *pause* rather than termination). Recommend one, with the breach
probabilities for each. Do not recommend switching programmes purely to avoid the
stop-loss requirement — the stop is protecting me.

---

## Working style

Start with Gate 0 and report before moving on. At each gate, tell me the number
and the verdict in a few lines — I do not need the narration. If you hit
something that changes the shape of the investigation, stop and say so rather
than working around it.

Unrelated but outstanding: the `.env` file with a real MT5 account number is
still in git history and needs purging with `git filter-repo` or BFG. Flag it if
you touch repo hygiene.
