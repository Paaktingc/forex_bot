# Uncommitted working-tree inventory (2026-07-29)

Snapshot taken on branch `the5ers-bootcamp` at HEAD `aba78dd` before starting
the FX currency-strength research. All items below are **prior-session work**.
They are preserved wholesale on backup branch
`backup/pre-currency-strength-working-tree-20260729` (explicit backup commit
`dc967f0`). Only this inventory and `provisional_risk_status.md` are committed
onto `the5ers-bootcamp`.

## ⛔ CRITICAL FINDING — committed baseline is NOT self-contained (Phase 0 blocker)

Cleaning the working tree to the committed HEAD does **not** yield a passing
baseline. The committed HEAD cannot even collect the test suite, because
**committed code depends on uncommitted files/attributes**:

- committed `main.py` (line 26) `import reconcile` — `reconcile.py` is **untracked**;
- committed `main.py` → `RiskManager.load_or_init(...)` — that method exists only
  in the **uncommitted** `risk_manager.py`;
- that uncommitted `risk_manager.py` references `config.OFFICIAL_DAILY_LOSS_PCT`
  and other `PROGRAMMES` constants that exist only in the **uncommitted**
  `config.py`;
- `main.py` also uses `bar_utils.drop_forming_bar` (`bar_utils.py` untracked).

So the reported "306 tests passing" is a property of the **full working tree**,
not of the committed HEAD. Restoring a green baseline requires re-introducing the
entire entangled Cycle-6 / deployment-Findings layer (`config.py`,
`risk_manager.py`, `reconcile.py`, `bar_utils.py`, `news_filter.py`,
`trade_journal.py`, `monte_carlo_dd.py`, `backtest.py`) — which this task
designates backup-only and forbids mixing into research commits.

**Resolution taken:** the working tree was returned to its original known-good
state (8 modified + untracked, `pytest` = 306 passed) so nothing is lost and the
repo is no more broken than received; everything is additionally preserved on the
backup branch. Per Phase-0 Step 0.4 ("do not begin Phase 1 if the clean committed
baseline fails"), **Phase 1 was NOT started.** Verdict:
`STOP — repository baseline could not be established`. A manual decision is
required (see "Recommended fix" at the bottom).

**Secret check:** no secret *values* are present in the working tree. Identifier
names matched in `brokers/ctrader_adapter.py` are env-based reads (no hardcoded
credentials); no `.env` is tracked or on disk. (A separate, previously-flagged
item — a real MT5 account number in *git history* — is a history-rewrite
remediation outside this Phase-0 working-tree scope and is not reintroduced here.)

`git diff --stat` (tracked, modified): README.md, backtest.py,
brokers/ctrader_adapter.py, config.py, monte_carlo_dd.py, news_filter.py,
risk_manager.py, trade_journal.py — 8 files, +618/−50.

| File | Status | Apparent purpose | Related line | Tests present | Recommended action |
|---|---|---|---|---|---|
| `config.py` | Modified | Cycle-6 PROGRAMME profiles (bootcamp/high_stakes) + provisional `risk_per_trade_pct=0.002` | The5ers programme layer (Cycle-6) | `test_programme_profiles.py`, `test_programme_risk_gate.py` | Preserve on backup branch only |
| `risk_manager.py` | Modified | Programme-aware sizing, profitable-day / official-daily-loss gates, drawdown breaker | Cycle-6 | `test_programme_profiles.py`, `test_deployment_safety.py` | Preserve on backup branch only |
| `monte_carlo_dd.py` | Modified | Multi-step programme Monte Carlo (Bootcamp 3-step / High Stakes 2-step) | Cycle-6 | `test_programme_profiles.py` | Preserve on backup branch only |
| `backtest.py` | Modified | Additional symbol-aware / bp-mode transfer-research changes beyond HEAD | Cycle-5/6 transfer | partial (`test_backtest.py`) | Requires manual decision |
| `news_filter.py` | Modified | News cache / per-symbol news currencies | Cycle-5/6 | — | Preserve on backup branch only |
| `trade_journal.py` | Modified | Broker-close reconciliation hooks | Finding 2 | `test_deployment_safety.py` | Preserve on backup branch only |
| `brokers/ctrader_adapter.py` | Modified | Drop still-forming trendbar (live) | Finding 6 | `test_deployment_safety.py` | Preserve on backup branch only |
| `README.md` | Modified | Doc updates for the above | Cycle-6 | — | Preserve on backup branch only |
| `bar_utils.py` | Untracked | `drop_forming_bar` helper | Finding 6 | `test_deployment_safety.py` | Preserve on backup branch only |
| `reconcile.py` | Untracked | Broker-side SL/TP close reconciliation | Finding 2 | `test_deployment_safety.py` | Preserve on backup branch only |
| `sleeve_model.py` | Untracked | Cycle-6 portfolio-sleeve barrier calibration model (research tool) | Cycle-6 | — | Preserve on backup branch only |
| `tests/test_deployment_safety.py` | Untracked | Regression tests for deployment Findings 1–7 | Findings 1–7 | (is a test) | Preserve on backup branch only |
| `tests/test_programme_profiles.py` | Untracked | Cycle-6 programme-profile layer tests | Cycle-6 | (is a test) | Preserve on backup branch only |
| `tests/test_programme_risk_gate.py` | Untracked | P(maxDD>5%)<5% risk-gate test (depends on invalidated leaky R distribution) | Step-1/Gate-A (invalidated line) | (is a test) | Requires manual decision |
| `research/xauusd_copier_teardown/**` | Untracked | XAUUSD copier reverse-engineering: report, pipeline scripts, reconstructed data/CSVs, chart | XAUUSD basket (retired) | — | Preserve on backup branch only |
| `PROMPT_pooled_frequency_study.md` | Untracked | Saved prompt text from an earlier session | n/a | — | Preserve on backup branch only (obsolete) |

## Notes on specific items

- **`config.py` risk default (`0.002`)**: conservative *provisional* default. Its
  original strategy-specific justification (the H1-regime feasibility envelope)
  was **invalidated** by the look-ahead-leakage finding. It is **not** an
  approved or optimal risk level and is **not** changed in this task. See
  `provisional_risk_status.md`.
- **`test_programme_risk_gate.py`** asserts the risk gate against a return
  distribution produced by the now-retired leaky H1 signal; it should not be
  landed on the research branch without revisiting that dependency — hence
  "requires manual decision", preserved on the backup branch.
- **`backtest.py`**: the committed HEAD already contains the symbol-aware engine;
  the uncommitted delta is additional transfer-research change and needs manual
  review before any landing.

All items are recoverable from
`backup/pre-currency-strength-working-tree-20260729` (`dc967f0`).

## Recommended fix (manual decision required, out of scope for this task)

The committed HEAD couples committed `main.py`/`tests/test_main.py` to the
uncommitted Cycle-6/deployment layer. To restore a clean, self-contained,
passing committed baseline, choose one:

1. **Land the load-bearing layer** — review and commit the coupled set
   (`config.py`, `risk_manager.py`, `reconcile.py`, `bar_utils.py`,
   `news_filter.py`, `trade_journal.py`, plus their tests
   `test_deployment_safety.py`, `test_programme_profiles.py`) as one reviewed
   "deployment hardening + programme layer" commit. Largest scope; makes HEAD
   green as-is. `test_programme_risk_gate.py` should be revisited separately
   (it depends on the invalidated leaky H1 return distribution).
2. **Decouple committed `main.py`** — guard/remove its `reconcile`,
   `bar_utils`, and `load_or_init` usage so the committed live-bot glue imports
   without the uncommitted modules. Smaller diff but changes live-bot code.

Both are repository-integrity decisions that belong to the maintainer, not to
this research task. Until one is done, `pytest` on a *clean* HEAD fails at
collection (`ModuleNotFoundError: reconcile`).
