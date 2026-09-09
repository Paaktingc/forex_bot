# XAUUSD-STDc Automated Bot — Reverse-Engineering & The5ers Bootcamp Suitability

**Account** 29498319 (POON KIT MEI) · **Symbol** XAUUSD-STDc · **Currency** USC (US cents)
**Statement period supplied** 29 June 2026 – 23 July 2026 · **Analysis date** 27 July 2026
**Evidence base** 16 daily MT5 statement sheets, 6,281,860 Dukascopy XAUUSD ticks (1–26 July 2026)

---

## 1. Executive conclusion

### What the bot most likely does

The account is a **copy-trading receiver** running an **unhedged, un-stopped, geometric-Martingale grid on XAUUSD**, managed as *baskets* that are closed collectively on a small profit measured from the basket's weighted-average entry.

The observable mechanism, in chronological order:

1. A basket opens with **0.02 lots** (the account minimum) — 92.8% of all 486 baskets.
2. If price moves **against** the position by roughly **2.1 USD/oz**, a further position is added in the same direction at the next rung of a fixed ladder: `0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.11, 0.14, 0.18, 0.23, 0.30, 0.39, 0.51, 0.67, 0.87`. 81.5% of all additions follow an adverse move.
3. The whole basket is closed in a single burst when price retraces to roughly **+2.3 USD/oz beyond the weighted-average entry** (median for winning baskets). 84% of baskets close within two timestamps.
4. **No stop-loss or take-profit is ever attached to any order.** All 2,721 orders are market orders with `S/L = 0` and `T/P = 0`. Exits are managed entirely by the (remote) master EA.

### What is known with confidence

| Fact | Confidence |
|---|---|
| Every one of 2,721 deals carries a comment `copy #<9–10 digit master ticket>` — this is a **copied receiver account**, not the originator | **Confirmed** |
| No stop-loss or take-profit on any position | **Confirmed** |
| Geometric Martingale ladder, ratio ≈ **1.30**, floor 0.02 lots — 88.1% of baskets reproduce the fitted table exactly | **Confirmed** |
| Grid additions are triggered by **adverse** movement (81.5%), median step **2.10 USD/oz**, non-expanding | **Strongly supported** |
| Basket exit at a small **fixed distance from weighted-average entry** (median 2.33 USD/oz for winners) | **Strongly supported** |
| Broker server clock = **UTC+3** (median price error vs Dukascopy 0.235 USD at that offset, 13.1 USD at UTC+0) | **Confirmed** |
| Contract size **1 troy ounce per lot**, P/L reported in **US cents**; leverage **≈ 1:500** | **Confirmed** (see §2.3) |
| Maximum hidden floating drawdown **−7,815 USC = −7.67% of balance** while closed-balance drawdown was **0.00%** | **Confirmed** (tick-reconstructed) |

### What remains unknown

- **The first-entry signal.** Not recoverable. No indicator rule tested reached usable precision (best: 2.61% vs a 1.01% random baseline). The trigger lives on the master account.
- **The exact basket-close rule.** The exit distance is *centred* on ~2.3 USD/oz but has a standard deviation of 1.25 — it is not a hard constant, so a trailing or adaptive component is likely.
- **Behaviour on the three missing statement days** (see §2.2), which together account for **−3,210.88 USC of losses that are absent from the supplied sheets**.
- Whether the 11.5 hours of simultaneous long+short exposure is deliberate hedging or two independent masters being copied into one account.

### How it makes money

It does not make money from directional prediction. It makes money by **converting a high probability of a small win into a small probability of a very large loss**. Over 17 trading days it produced:

- **486 baskets, 90.3% win rate, zero losing days in the supplied sheets, 0.00% closed-balance drawdown**
- Median basket profit **+9.29 USC (US$0.09)**; largest basket loss **−882 USC**
- Meanwhile the worst *unrealised* position was **−8,465 USC on a single basket** — **911× the median win**

That ratio is the whole strategy. The equity curve in Chart 10 shows the balance line rising smoothly while the equity line repeatedly stabs downward.

### Main hidden risks

1. **Unbounded loss.** With no stop-loss and a 1.30 ladder, loss grows super-linearly with adverse distance. At 16 levels a 60 USD/oz adverse move produces **−12,954 USC (−12.7% of balance)**; a 140 USD move produces **−42,554 USC (−41.7%)**. Gold moved 243 USD (6.1%) inside the 26 days of tick data supplied.
2. **The credit facility masks the risk.** A 50,000 USC credit bonus inflates equity by 50%, so margin never binds at 1:500. On any prop account there is no such cushion.
3. **The supplied statement is survivorship-filtered.** Three ledger discontinuities show −3,210.88 USC of losses on days whose statements were not provided. Real net was **+12,738.36 USC**, not the **+15,949.24** the supplied trades imply — a **20% overstatement**.
4. **Copier fragility.** 18.2% of additions arrive in sub-5-second bursts (up to 12 orders in 2 seconds). A single missed or rejected copy leaves the receiver with an unmanaged ladder and no exit logic of its own.

### Suitability for The5ers Bootcamp

**Not suitable, on five independent grounds** — see the compliance matrix in §9. The decisive one is that Bootcamp requires a stop-loss on *every* position and this bot places none, ever.

---

## 2. Data-quality report

### 2.1 What was actually supplied vs what the brief requested

| Requested | Supplied | Status |
|---|---|---|
| MT4/MT5 trading statement | `xauusd trading lot.xlsx` — 16 daily sheets | ✅ present |
| Same-broker XAUUSD M1 data | — | ❌ **missing** |
| Same-broker bid/ask tick data | — | ❌ **missing** |
| Dukascopy XAUUSD M1 | — | ❌ **missing** (download script only) |
| Dukascopy XAUUSD tick | `xauusd-tick-2026-07-01-...csv` | ⚠️ present, **starts 1 July** (period starts 29 June) |
| Broker symbol specification | `account_and_timezone.txt` | ❌ **template is entirely blank** |
| Account balance / leverage / currency / timezone | `account_and_timezone.txt` | ❌ **template is entirely blank** |
| The5ers Bootcamp rules | — | ❌ retrieved from the5ers.com instead |

Every field in `account_and_timezone.txt` is empty. All symbol-specification values in this report were therefore **derived from the statement arithmetic and validated numerically** (§2.3), not read from the broker. They should be confirmed against MT5 before any code is written.

**M1 bars used in this report were synthesised from the Dukascopy tick file**, not exported from the broker. Broker prices differ from Dukascopy by a median 0.235 USD.

### 2.2 Statement completeness — three unexplained ledger discontinuities

Sheets are one per trading day. Walking `Previous Ledger Balance → Balance`:

| Date | Prev. ledger | Expected | Gap | Note |
|---|---|---|---|---|
| 7 Jul (Sheet11) | 102,495.92 | 103,869.58 | **−1,373.66** | **6 July (Mon) statement missing** |
| 15 Jul (Sheet7) | 105,488.74 | 105,945.68 | **−456.94** | **13–14 July (Mon/Tue) missing** |
| 20 Jul (Sheet4) | 105,320.51 | 106,700.79 | **−1,380.28** | no missing weekday — unexplained |
| | | **Total** | **−3,210.88** | |

Corroborating evidence: nine `out` deals at **2026-07-07 02:57:46** close positions that were never opened in any supplied sheet — they belong to the missing 6 July session.

**Consequence:** the supplied sheets show only profitable days. The three missing/unexplained sessions were collectively **losing**. Any performance figure computed from the supplied trades alone is biased upward by ~25%.

**This data was not repaired.** All reported trade statistics are computed on the 1,356 fully-reconstructable round trips; the −3,210.88 is reported separately and never silently absorbed.

### 2.3 Symbol and account specification — derived, not supplied

| Field | Value | How established |
|---|---|---|
| Symbol | `XAUUSD-STDc` | statement, all 2,917 rows |
| Platform | MT5, **hedging** mode | separate `Deals`/`Positions` sections; `Entry: in/out`; 11.5 h of simultaneous long+short |
| Account currency | **USC** (US cents), 100 USC = US$1 | statement header + deposit comment |
| Digits / point | **2 / 0.01** | all 2,721 prices have 2 decimals |
| **Contract size** | **1 troy ounce per lot** | P/L identity `profit_USC = Δprice × lots × 100` reproduces the implied entry price of **1,356 / 1,356 (100.00%)** round trips to **0.00000 USD error** |
| Tick value | 1.00 USC per 0.01 lot per 0.01 USD move | same identity |
| Min / step / max lot | 0.02 / 0.01 / ≥3.34 observed | smallest and largest observed; broker max not observable |
| **Leverage** | **≈ 1:500** | 22 Jul EOD: 2.80 lots × 4,130.29 = US$11,564.81 notional vs `Margin Requirements` 2,315.84 USC = US$23.16 → **499.4:1** |
| Commission | **0.00** | all 2,721 deals |
| Fee | **0.00** | all 2,721 deals |
| Swap | **+226.66 USC total** (net credit; short gold earns) | e.g. +44.52 USC on 0.55 lots held overnight |
| Starting balance | **100,000.00 USC = US$1,000** | deposit 2026-06-28 17:10:21, `Deposit-CC-AP-Payabl-CPS` |
| **Credit facility** | **50,000.00 USC = US$500 bonus** | `Credit In-GCN5025`, 2026-06-28 17:10:22, constant all 16 sheets |
| Final balance | 112,738.36 USC (equity 162,738.36 incl. credit) | Sheet1 |
| Margin call / stop-out | **not observable** | never approached; must be confirmed with broker |
| Server timezone | **UTC+3** | grid search over offsets, median \|statement price − Dukascopy mid\| = **0.235 USD at +3** vs 13.06 at +0 |
| UK conversion | UK (BST) = **server − 2 h** | UTC+3 → UTC+1 |

### 2.4 Statement integrity checks

| Check | Result |
|---|---|
| Order rows / deal rows | 2,917 / 2,919 |
| Non-trade operations | 2 (deposit, credit) — correctly excluded |
| **Duplicate deal tickets** | **196**, all in Sheet15 (30 Jun) which re-states Sheet16 (29 Jun) cumulatively. All duplicates **byte-identical** on every trade field → safe to deduplicate. **2,721 unique deals** remain |
| Timestamp ordering | monotonic after sort; no impossible orderings |
| Impossible prices | none (all 3,959–4,203, consistent with tick range) |
| Order state | `filled` on 100% — **no cancelled or rejected orders in the sample** |
| Partial fills | **0** (`requested == filled` on every order) |
| Partial closes | **0** — every position closed in one deal |
| SL / TP set | **0 / 2,721** |
| Order price type | `market` on 100% |
| Request→fill latency | median 0 s, p95 1 s, max 4 s |
| Round-trip reconciliation | **exact on 15 of 16 days**; 7 July differs by −95.20 USC (the six-July orphans) |

### 2.5 Tick-data integrity (Dukascopy)

| Check | Result |
|---|---|
| Rows / span | 6,281,860 · 2026-07-01 00:00:00 UTC → 2026-07-26 22:59:56 UTC |
| Monotonic / duplicates | monotonic ✅ · 0 duplicates ✅ |
| Missing bid or ask | 0 |
| Crossed or zero spread | 0 |
| Gaps > 1 h | 18 — all daily 22:00–23:00 UTC rollovers and three weekends (49–53 h). No unexpected gaps |
| Spread | median **0.68 USD**, p90 0.82, p99 0.97, max 15.00 |
| **Coverage gap** | **29–30 June has no tick data.** 44 of 486 baskets (9.1%) could not have MAE/MFE computed |

### 2.6 Assumptions carried forward

1. Contract multiplier of 100 USC per lot per USD — *validated to zero error on 1,356 trades*, but confirm `TickValue` in MT5.
2. Position matching used lot-size + implied-entry-price + FIFO. Because implied entry matched actual entry to 0.00000 on every trade, this matching is **exact, not approximate**.
3. Dukascopy prices are a *proxy* for the broker feed (median offset 0.235 USD). MAE/MFE figures therefore carry roughly ±0.25 USD/oz of feed uncertainty — negligible relative to the 8,465 USC worst case.
4. Basket definition: a maximal run in which same-direction exposure never returns to zero. Alternative definitions (copy-ID contiguity, burst clustering) were tested and gave the same ladder and exit conclusions.

---

## 3. Clean transaction dataset

**`trades_reconstructed.csv`** — 1,356 rows, one per fully-matched round-trip position.

Columns: `pos_id, dir, open_dt, open_price, lots, close_dt, close_price, implied_entry, price_match_err, partial_open, partial_close, profit, swap, comm, fee, open_deal, close_deal, open_order, close_order, open_copy, close_copy, open_lat, close_lat, duration_s, net, pips, basket`

Also provided: `raw_orders.csv`, `raw_deals.csv`, `raw_positions.csv`, `raw_summary.csv` (unmodified section dumps, one row per statement line, with `_sheet` provenance).

Sample (first basket of 23 July, Sheet1 rows 115–126 / 127–138):

| pos_id | dir | open_dt | open_price | lots | close_dt | close_price | profit | basket |
|---|---|---|---|---|---|---|---|---|
| 619886277 | long | 2026-07-23 02:27:50 | 4120.48 | 0.02 | 2026-07-23 02:33:19 | 4122.24 | 4.02 | L245 |
| 619886505 | long | 2026-07-23 02:27:50 | 4120.43 | 0.03 | 2026-07-23 02:33:22 | 4122.26 | 5.49 | L245 |
| 619886735 | long | 2026-07-23 02:27:51 | 4120.39 | 0.04 | 2026-07-23 02:33:21 | 4122.26 | 7.48 | L245 |

---

## 4. Basket dataset

**`baskets_reconstructed.csv`** and **`baskets_with_mae.csv`** — 486 rows.

Columns: `basket, dir, start, end, duration_s, n_entries, total_lots, max_simul_lots, wavg_entry, wavg_exit, first_entry, last_entry, entry_span, gross_profit, swap, comm, net, profit_per_001lot, exit_dist_from_wae, n_close_events, all_closed_together, med_gap_s, max_gap_s, med_price_step, n_partial, outcome, mae_usd, mfe_usd, min_float, max_float, covered`

Summary:

| Metric | Value |
|---|---|
| Baskets | 486 (245 long, 241 short) |
| Entries per basket | min 1 · median 2 · p90 6 · **max 16** |
| Single-entry baskets | 228 (46.9%) |
| Total lots per basket | median 0.04 · p99 2.55 · **max 14.20** |
| Duration | median 346 s · p90 2,226 s · **max 35,762 s (9.9 h)** |
| Baskets held across a date boundary | 1 (S241, 22→23 July, 9.9 h) |
| Baskets held across a weekend | 0 |

---

## 5. Evidence and confidence table

| # | Inferred rule | Classification | Evidence |
|---|---|---|---|
| 1 | Account is a **copy receiver**, not the master | **Confirmed** | 2,721/2,721 deals comment `copy #…`; Sheet1 rows 4–96, Sheet16 rows 4–…; master IDs 944,609,691 → 1,010,800,913 strictly increasing with local time |
| 2 | **No stop-loss, no take-profit** on any position | **Confirmed** | `S / L = 0` and `T / P = 0` on all 2,917 Orders rows; `Positions` section 22 Jul (Sheet2 rows 121–135) also shows `0 / 0` |
| 3 | All entries are **market orders** | **Confirmed** | `Price = market`, `State = filled`, 2,917/2,917 |
| 4 | Contract = 1 oz/lot, P/L in cents | **Confirmed** | Deal 481141972 (Sheet1 r100): buy-out 0.55 @ 4122.14, profit 1,641.20 → implied entry 4151.98 = exact open price of position 619054064 (Sheet2 r134). 1,356/1,356 exact |
| 5 | Base lot = **0.02** (account minimum) | **Confirmed** | 451/486 baskets (92.8%) open at 0.02; 723/1,356 positions are 0.02 |
| 6 | Ladder = `0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.11,0.14,0.18,0.23,0.30,0.39,0.51,0.67,0.87` | **Confirmed** | Modal lot at every level 0–15 matches exactly; **428/486 baskets (88.1%)** reproduce the whole table; e.g. basket L184 (Sheet6, 16 Jul 15:33:01 → 16:11:12) is a perfect 16-level instance |
| 7 | Ladder ratio ≈ **1.30**, `next = round(prev × 1.30, 2)` | **Strongly supported** | 405/451 **ascending** burst transitions (89.8%) match exactly; log-linear fit over levels 5–15 gives r = 1.306. Fails at 0.08→0.11 and 0.51→0.67, so the lookup table is the better representation than pure recursion |
| 8 | Additions triggered by **adverse** movement | **Strongly supported** | 81.5% of 870 transitions follow an adverse move; only 6.7% follow a favourable one |
| 9 | Grid step ≈ **2.1 USD/oz**, **non-expanding** | **Strongly supported** | n=712 spaced additions: median 2.105, IQR 1.42–3.10. Median by level: 2.14, 2.15, 2.05, 2.23, 2.02, 1.62, 2.78 … no monotonic trend |
| 10 | Basket closes at ≈ **+2.3 USD/oz from weighted-average entry** | **Strongly supported** | 439 winning baskets: median 2.329, IQR 1.775–3.013. Lowest-variance candidate of the three tested (CV 0.94 vs 6.37 for fixed monetary profit) |
| 11 | Basket closed **all-at-once** | **Strongly supported** | 84% of baskets close inside ≤2 distinct timestamps |
| 12 | Server clock = **UTC+3** | **Confirmed** | offset grid search, error 0.235 USD at +3 vs ≥7.24 at every other integer offset |
| 13 | **Counter-trend** direction bias | **Plausible** | Longs open after a median −4.16 bp 5-min move, shorts after +4.67 bp (Mann-Whitney p < 0.0001); Bollinger position 0.385 (long) vs 0.613 (short), p < 0.0001. Effect is real but far too weak to time entries |
| 14 | Activity concentrated **07:00–20:00 server (05:00–18:00 UK)** | **Strongly supported** | **94.1%** of entries; peak hour 16:00 server (14:00 UK, London–NY overlap) with 222 entries; 15:00 and 17:00 next with 160 each |
| 15 | **Tick/event-driven**, not candle-driven | **Strongly supported** | 4.6% of entries in seconds 0–2 of a minute vs 5.0% uniform expectation — no candle-boundary clustering. Sub-second bursts of up to 12 orders |
| 16 | No balance- or equity-scaled sizing | **Strongly supported** | Balance rose 100,000 → 112,738 USC (+12.7%) with **no change** to the 0.02 base lot or the ladder |
| 17 | Genuine simultaneous hedging occurs | **Plausible** | 41,246 s (11.5 h) with both directions open; 18.5% of baskets start before the previous one ends. Could be deliberate hedging or two masters |
| 18 | Duplicate parallel copies | **Plausible** | e.g. Sheet2 r134/r135: two identical 0.55 sells @ 4151.98 one second apart; Sheet1 r4–11 and r12–18 close two identical ladders |
| 19 | Emergency / recovery mode exists | **Weak evidence** | 25 descending burst sequences (e.g. basket S241 opens 0.34→0.02, S019 opens 0.70→0.09) vs 450 ascending. Mechanism not identifiable |
| 20 | Fixed monetary basket target | **Contradicted** | Winner net P/L ranges 0.06 → 4,021.16 USC, CV 6.37 |
| 21 | Fibonacci ladder | **Contradicted** | Only 19.2% of ascending transitions fit r = 1.618 vs 89.8% for r = 1.30. The Fibonacci-looking values (0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34) appear only in the 25 descending sequences |
| 22 | **First-entry timing signal** | **Unknown** | Best of 8 candidate rules: 2.61% precision at 29.0% recall, against a 1.01% base rate. Not usable |
| 23 | Basket exit is a hard constant | **Contradicted** | σ = 1.25 USD around a 2.33 median; adaptive or trailing component likely |

---

## 6. Original strategy specification (most likely behaviour)

```
ON  new-basket signal from master (mechanism unknown, weak counter-trend tilt)
    open market order, direction = signal, lots = 0.02
    no stop-loss, no take-profit
    level := 0

WHILE basket open
    IF price has moved ADVERSELY ~2.1 USD/oz from the last entry
       AND level < 15
    THEN
       level := level + 1
       lots  := LADDER[level]                 # 1.30 geometric, floor 0.02
       open market order, same direction, no SL, no TP
       (the copier may deliver several levels within the same second)

    wae := Σ(entry_price × lots) / Σ(lots)

    IF price >= wae + ~2.3 USD/oz   (long)
       OR price <= wae − ~2.3 USD/oz (short)
    THEN
       close ALL positions in the basket in one burst
       basket ends

    # there is NO loss-side exit of any kind

AFTER close
    43.6% of baskets restart within 60 s; 18.5% overlap the previous basket
```

`LADDER = [0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.11, 0.14, 0.18, 0.23, 0.30, 0.39, 0.51, 0.67, 0.87]`
Cumulative exposure if all 16 levels fill: **3.70 lots** (= 3.70 oz ≈ US$15,200 notional).

---

## 7. Parameter table

| Parameter | Best estimate | Range | Confidence |
|---|---|---|---|
| Initial lot | 0.02 | fixed (= min lot) | **Confirmed** (92.8%) |
| Lot sequence | `0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.11,0.14,0.18,0.23,0.30,0.39,0.51,0.67,0.87` | as listed | **Confirmed** (88.1% exact) |
| Lot multiplier | 1.30 | 1.28 – 1.32 | **Strongly supported** (89.8% of ascending steps) |
| Rounding | round-half-even to 0.01 | — | **Plausible** (explains 0.05→0.06; fails 0.08→0.11) |
| Grid distance | 2.1 USD/oz | 1.4 – 3.1 (IQR) | **Strongly supported** |
| Volatility multiplier | none detected | — | **Plausible** — step/ATR ratio is unstable (median ATR14 at entry 2.5–2.7, similar magnitude, so an ATR≈1.0× rule cannot be excluded) |
| Maximum levels | 16 observed | ≥16; no cap proven | **Weak** — 16 is the sample maximum, not a demonstrated limit |
| Max total exposure | 4.30 lots (ladder) / **14.20 lots observed** (basket S019, 1 Jul) | — | **Confirmed** for the observation |
| Basket target | +2.3 USD/oz from WAE | 1.8 – 3.0 (IQR) | **Strongly supported** |
| Stop conditions | **none** | — | **Confirmed** |
| Active hours | 07:00–20:00 server (05:00–18:00 UK), 94.1% of entries; peak 16:00 server | — | **Strongly supported** |
| Restart delay | median 96 s | 43.6% < 60 s; 18.5% overlap | **Strongly supported** |
| Commission | 0 | — | **Confirmed** |
| Swap | net +226.66 USC over the period | — | **Confirmed** |

---

## 8. Risk and hidden drawdown

### 8.1 Realised (what the statement shows)

| Metric | Value |
|---|---|
| Baskets | 486 |
| Win rate | **90.3%** (439 win / 47 loss) |
| Gross profit / gross loss | +17,269.54 / −1,415.50 USC |
| Average win / average loss | +39.34 / −30.12 USC (payoff 1.31) |
| Profit factor | **12.20** |
| Expectancy | +32.62 USC / basket |
| Median basket profit | **+9.29 USC (US$0.09)** |
| Largest win / loss | +4,021.16 / **−882.44** USC |
| Losing days in supplied sheets | **0 of 16** |
| **Closed-balance max drawdown** | **0.00%** |

### 8.2 Hidden (what the ticks show)

| Metric | Value | % of 101,942 USC balance |
|---|---|---|
| Worst single-basket floating loss | **−8,465.21 USC** (S241, 22 Jul) | **−8.30%** |
| 2nd worst | −7,733.45 USC (S019, 1 Jul, 14.20 lots) | −7.59% |
| 3rd worst | −4,493.18 USC (L184, 16 Jul) | −4.41% |
| **Worst account-level simultaneous floating** | **−7,814.97 USC** | **−7.67%** (−5.14% of equity incl. the 50,000 credit) |
| Median floating loss ÷ realised profit, winners | **1.11×** | — |

The account carried, on median, **more unrealised loss than the profit it eventually booked** on every winning basket. Chart 8 shows the pattern clearly: a flat line punctuated by deep spikes.

### 8.3 Stress tests (analytic, on the fitted 16-level ladder)

Floating loss in USC for an adverse move of *X* USD/oz with a 2.1 USD grid step:

| Adverse move | 8 levels | 12 levels | 16 levels | 16 levels as % of 101,942 USC |
|---|---|---|---|---|
| 1% (41 USD) | −1,272 | −3,005 | −5,924 | **−5.8%** |
| 2% (82 USD) | −2,952 | −8,171 | −21,094 | **−20.7%** |
| 3% (123 USD) | −4,634 | −13,337 | −36,264 | **−35.6%** |

Even a **1% adverse intraday move already exceeds the 5% Bootcamp limit** if the ladder is allowed to run to 16 levels.

| Scenario | Effect |
|---|---|
| **Sustained trend** (gold moved 3,959→4,203, +6.1%, inside the tick window) | Ladder exhausts; loss compounds without exit. This is the terminal failure mode |
| **Widened spread** (max observed 15.00 USD vs 0.68 median) | Adds 22× the normal cost per round trip; on a 0.02-lot basket targeting 2.3 USD that alone can turn the trade negative |
| **Slippage** | 0 observed (latency ≤4 s), but sub-5-second bursts of 12 orders are precisely where slippage concentrates |
| **Missed / rejected grid order** | 0 observed in 2,721. A miss breaks the weighted-average maths and the basket may never reach its target |
| **Delayed copy execution** | Median latency 0 s here, but entries are triggered by 2.1 USD moves — a 3–5 s delay in fast gold consumes an entire grid step |
| **Overnight gap** | 5 baskets held across a date boundary; the 49–53 h weekend gaps in the tick file are where an unhedged 4.3-lot ladder is uninsurable |
| **Major news** | No filter observed; entries occur throughout US data windows |
| **Disconnection** | Fatal. Receiver has no local exit logic — positions have no SL and the exit is issued by the master |
| **Margin call / risk of ruin** | Never approached at 1:500 **with a 50% credit bonus**. At Bootcamp's **1:30**, the 4.30-lot ladder needs ~US$590 of margin per basket — but more importantly the loss reaches the 5% limit long before margin binds |

---

## 9. The5ers Bootcamp compliance matrix

Rules as published on the5ers.com (page updated 25 June 2026). **Anything marked "Requires confirmation" should be put to The5ers in writing before spending money.**

| # | Area | Bootcamp rule | Original bot | Verdict |
|---|---|---|---|---|
| 1 | **Stop-loss** | Mandatory SL on **every** position, no exceptions. Opening a position without one = a violation; **5 violations terminate the account** | 0 stop-losses on 2,721 orders | **FAIL** — 2,721 violations |
| 2 | **SL size** | SL may not risk >2% of balance on a single position | n/a (no SL) | **FAIL** |
| 3 | **External copy trading** | External signal copying **prohibited** | 100% of deals are `copy #…` from an external master | **FAIL** |
| 4 | **Internal copy trading** | Permitted on other programmes, **excluded on Bootcamp** | — | **FAIL** if replicated across accounts |
| 5 | **EA usage** | Permitted | EA-driven | **PASS** |
| 6 | **Tick scalping / latency arbitrage** | Prohibited | Median basket 346 s; 46.9% single-entry with a ~2.4 USD target. Not latency arbitrage, but sub-second 12-order bursts and small targets invite scrutiny | **UNCLEAR — requires written confirmation** |
| 7 | **HFT** | Prohibited | 2,721 orders in 17 days = 160/day, bursts of 12/second | **UNCLEAR — requires written confirmation** |
| 8 | **Arbitrage** | Prohibited | No evidence | **PASS** |
| 9 | **Max loss** | 5% per evaluation step (4% funded) | Tick-reconstructed floating drawdown **−7.67%** | **FAIL** |
| 10 | **Daily rule** | 3% daily pause, **funded stage only** — not in evaluation | Worst single-basket floating −8.30% would breach the funded daily rule | **FAIL at funded stage** |
| 11 | **Profit target** | 6% per step ×3, then 5% funded | +12.7% booked in 17 days (but on 1:500 with a credit bonus) | **PASS on paper only** |
| 12 | **Leverage** | **1:30**, all steps | Operating at **1:500** | **FAIL** — 16.7× reduction; the ladder must be re-derived from scratch |
| 13 | **XAUUSD margin** | 1:30 → ~3.33% of notional | 3.70-lot ladder ≈ US$15,200 notional → ~US$507 margin | **PASS** on a ≥US$20k account, but see #9 |
| 14 | **News trading** | Permitted on Bootcamp (excl. bracket strategies) | No news filter, but no bracketing either | **PASS** |
| 15 | **Hedging** | Permitted, **with SL on every position** | 11.5 h of simultaneous long+short, no SL | **FAIL** (via #1) |
| 16 | **Overnight / weekend holds** | Permitted | 5 overnight, 0 weekend | **PASS** |
| 17 | **Minimum profitable days** | None in Bootcamp evaluation | n/a | **PASS** |
| 18 | **Inactivity** | Account closes after 30 consecutive inactive days | 160 orders/day | **PASS** |
| 19 | **Credit/bonus equity** | Not available on prop accounts | Strategy sized against 150,000 USC equity, only 100,000 of which was real | **FAIL** — economics do not transfer |

**Result: 6 hard FAILs, 2 UNCLEAR, 11 PASS.** Any single one of #1, #3, #9 or #12 is disqualifying on its own.

---

## 10. Independent modified strategy specification

This is a **new strategy**, not a re-parameterisation of the original. It shares only the two statistically supported structural ideas — *scale into adverse movement on a bounded ladder* and *exit the basket on weighted-average-entry retracement* — both of which are generic, publicly documented techniques. It uses **no copied signal**, and every rule below is independently computable from price alone.

### 10.1 Hard constraints (non-negotiable)

| Control | Setting |
|---|---|
| Signal source | **Locally generated. No copier, no external feed, ever.** |
| Risk per basket | **0.25%** of starting balance (research band 0.20–0.35%) |
| Max additions | **3** (4 positions total) — research band 2–4 |
| Ladder | **arithmetic**, `1.0 / 1.3 / 1.6 / 1.9 ×` base — **no geometric Martingale** |
| Max total exposure | `4 × base_lot × 1.45`, hard-capped at 1.0% of balance at the basket stop |
| Visible stop-loss | **On every position**, set to the basket stop level, attached in the same `OrderSend` |
| Basket stop | `WAE ∓ 2.0 × ATR(H1)`, never wider than the 0.25% risk budget |
| Internal daily stop | **−0.75%** of starting balance (band 0.75–1.00%) → flat + halt for the session |
| Internal total DD stop | **−2.0%** of starting balance (band 2.0–2.5%) → flat + halt permanently, human review |
| Spread filter | No entry if spread > **1.20 USD** (≈ p99.5 of observed) or > 2.0× the 60-min median |
| Volatility filter | No entry if ATR(M15) outside the 20th–90th percentile of its trailing 30-day distribution |
| News filter | Optional; if enabled, no new basket ±15 min around high-impact USD/XAU events |
| Max basket duration | **4 hours** → close at market regardless of P/L |
| Cooldown | After any losing basket: **60 min**. After 2 consecutive losers: **rest of session** |
| Session filter | 07:00–19:00 server (UTC+3) = 05:00–17:00 UK. No Friday entries after 17:00 server |
| Weekend | Flat by Friday 20:00 server. No exceptions |
| Rejected order | If an addition is rejected, **do not retry**; recompute WAE, tighten the basket stop to the already-committed risk, and mark the basket `degraded` (exit-only) |
| Connection loss | Server-side SL on every position is the primary failsafe. On reconnect, reconcile positions before any new order |
| Emergency close | Single function, callable manually and by the daily/total DD monitors, that closes largest-loss-first and cancels pending orders |

### 10.2 Position sizing — derived, not inherited

```
risk_money   = balance × risk_pct                      # e.g. 20,000 × 0.0025 = $50
stop_dist    = 2.0 × ATR_H1                            # USD per oz
ladder_w     = [1.00, 1.30, 1.60, 1.90]                # weights, sums to 5.80
# worst case: all 4 fill, basket stopped at WAE − stop_dist
# loss ≈ Σ(w_i) × base_lot × contract × stop_dist
base_lot     = risk_money / (Σ(ladder_w) × contract_size × stop_dist)
base_lot     = floor_to_step(base_lot, lot_step)
assert base_lot >= min_lot                             # else SKIP the trade
assert margin_for(Σ(ladder_w) × base_lot) < 0.25 × free_margin
```

Worked example — US$20,000 Bootcamp Step 1, 1:30, ATR_H1 = 6.0 USD, standard 100 oz contract:

```
risk_money = 50.00 ;  stop_dist = 12.00
base_lot   = 50 / (5.80 × 100 × 12.00) = 0.00718  →  0.01 lots (rounded to min)
            → actual worst-case loss = 5.80 × 0.01 × 100 × 12 = $69.60 = 0.35% of balance
```
0.35% exceeds the 0.25% target, so the rule **reduces the ladder to 3 rungs (1.0/1.3/1.6 = 3.9)** → $46.80 = **0.234%** ✅. This is exactly the kind of constraint the original bot never applies.

### 10.3 Independent first-entry signal

Because the master's trigger is unrecoverable, a **new** signal is required. The only statistically supported hint from the data is a weak counter-trend tilt (§5 rule 13), so the proposed starting point — to be *validated, not assumed* — is:

```
regime  : ATR(M15) within its 20th–90th trailing percentile      (avoid dead and chaotic markets)
context : |close − EMA200(M15)| < 1.5 × ATR(H1)                  (no entries far from value)
trigger : Bollinger(M15, 20, 2.0) close outside the band
          AND RSI(M15,14) < 30 (long) or > 70 (short)
          AND the next M15 bar closes back inside the band       (rejection confirmation)
filters : spread, session, news, cooldown  (§10.1)
```

This is a well-known mean-reversion template, **not** a reconstruction of the master. It must clear the validation gates in §12 before it is trusted.

### 10.4 Basket management

```
entry 0: base_lot,  SL = entry ∓ stop_dist
add   i: only if price moved 0.8 × ATR_H1 adversely from the previous entry
         AND i < max_additions
         AND the regime/spread filters still pass
         lots = base_lot × ladder_w[i]
         then RESET every position's SL to  WAE ∓ stop_dist   (the shared basket stop)
exit  : take-profit  = WAE ± 0.6 × ATR_H1
        OR basket stop hit (server-side, on every ticket)
        OR 4 hours elapsed
        OR daily/total DD monitor fires
```

Note the critical inversion versus the original: **the stop distance is fixed at the outset and the ladder is sized to fit inside it**, rather than the ladder being fixed and the loss allowed to grow.

---

## 11. Pseudocode

```
# ─────────────────── CONFIG ───────────────────
RISK_PCT=0.0025  DAILY_STOP=0.0075  TOTAL_STOP=0.020
MAX_ADD=3  LADDER=[1.00,1.30,1.60,1.90]
STOP_ATR=2.0  TP_ATR=0.6  GRID_ATR=0.8
MAX_SPREAD=1.20  MAX_DUR=4h  COOLDOWN=60min
SESSION=(07:00,19:00) server(UTC+3)

# ─────────────────── STATE ───────────────────
basket = None ; day_pl = 0 ; peak_equity = balance
halted_day = False ; halted_total = False ; consec_losses = 0 ; cooldown_until = 0

# ─────────────────── MAIN LOOP (on each completed M15 bar) ───────────────────
on_bar():
    if not reconcile_positions_with_broker(): return          # failsafe: connection
    update_indicators()
    if drawdown_monitor(): return                              # may emergency_close()
    if basket is None: try_open_basket()
    else:              manage_basket()

# ─────────────────── SIGNAL ───────────────────
signal():
    if not in_session(now) or halted_day or halted_total:   return NONE
    if now < cooldown_until:                                return NONE
    if spread() > MAX_SPREAD or spread() > 2*median_spread_60m(): return NONE
    if not pctile20 <= ATR_M15 <= pctile90:                 return NONE
    if news_filter_on and near_high_impact_event(15min):    return NONE
    if abs(close - EMA200_M15) > 1.5*ATR_H1:                return NONE
    if prev_close < BB_lower and close > BB_lower and RSI14 < 30: return LONG
    if prev_close > BB_upper and close < BB_upper and RSI14 > 70: return SHORT
    return NONE

# ─────────────────── SIZING ───────────────────
compute_base_lot(dir):
    risk   = balance * RISK_PCT
    stop_d = STOP_ATR * ATR_H1
    for n in [MAX_ADD+1, MAX_ADD, ..., 1]:                     # shrink ladder to fit risk
        w   = sum(LADDER[:n])
        lot = floor_step(risk / (w * CONTRACT * stop_d), LOT_STEP)
        if lot >= MIN_LOT and w*lot*CONTRACT*stop_d <= risk*1.05:
            if margin_required(w*lot) < 0.25*free_margin: return lot, n, stop_d
    return None                                                # SKIP: cannot size safely

# ─────────────────── FIRST ENTRY ───────────────────
try_open_basket():
    d = signal();  if d == NONE: return
    sized = compute_base_lot(d);  if sized is None: return
    lot, n_rungs, stop_d = sized
    sl = price - stop_d if d==LONG else price + stop_d
    tk = send_market_order(d, lot, sl=sl, tp=NONE)             # SL ALWAYS ATTACHED
    if tk is FAILED: log_and_abort(); return
    basket = Basket(dir=d, tickets=[tk], lots=[lot], prices=[fill], stop_d=stop_d,
                    max_rungs=n_rungs, opened=now, degraded=False)
    set_basket_protection()

# ─────────────────── ADDITIONAL ENTRIES ───────────────────
manage_basket():
    b = basket
    if now - b.opened > MAX_DUR:            close_basket("timeout");  return
    wae = weighted_average_entry(b)
    if hit_take_profit(wae, b):             close_basket("target");   return
    if len(b.tickets) < b.max_rungs and not b.degraded:
        adverse = (b.prices[-1] - price) if b.dir==LONG else (price - b.prices[-1])
        if adverse >= GRID_ATR * ATR_H1 and spread() <= MAX_SPREAD:
            lot = round_step(b.lots[0] * LADDER[len(b.tickets)], LOT_STEP)
            sl  = wae_after_add_stop(b, lot)
            tk  = send_market_order(b.dir, lot, sl=sl, tp=NONE)
            if tk is FAILED:
                b.degraded = True                              # FAILSAFE: no retry
                log("addition rejected -> basket exit-only")
            else:
                b.tickets.append(tk); b.lots.append(lot); b.prices.append(fill)
            set_basket_protection()

# ─────────────────── WEIGHTED AVERAGE + SHARED STOP ───────────────────
weighted_average_entry(b):  return sum(p*l for p,l in zip(b.prices,b.lots)) / sum(b.lots)

set_basket_protection():
    b = basket ; wae = weighted_average_entry(b)
    sl = wae - b.stop_d if b.dir==LONG else wae + b.stop_d
    tp = wae + TP_ATR*ATR_H1 if b.dir==LONG else wae - TP_ATR*ATR_H1
    # cap: never let the shared stop exceed the risk budget
    if projected_loss_at(sl) > balance*RISK_PCT:
        sl = tighten_to_budget(balance*RISK_PCT)
    for tk in b.tickets: modify_order(tk, sl=sl, tp=tp)        # VISIBLE ON EVERY TICKET

# ─────────────────── DRAWDOWN PROTECTION ───────────────────
drawdown_monitor():
    eq = balance + floating_pl()
    peak_equity = max(peak_equity, eq)
    if day_pl + floating_pl() <= -DAILY_STOP*start_balance:
        emergency_close("daily stop"); halted_day = True; return True
    if (eq - start_balance) <= -TOTAL_STOP*start_balance:
        emergency_close("total stop"); halted_total = True; alert_human(); return True
    return False

# ─────────────────── EXITS, RESTART, FAILSAFE ───────────────────
close_basket(reason):
    for tk in sorted(basket.tickets, key=unrealised_pl):       # worst first
        close_at_market(tk)
    realised = sum_realised(basket); day_pl += realised
    consec_losses = consec_losses+1 if realised < 0 else 0
    if realised < 0: cooldown_until = now + COOLDOWN
    if consec_losses >= 2: halted_day = True
    basket = None

emergency_close(reason):
    cancel_all_pending(); close_basket(reason); log_critical(reason); notify()

reconcile_positions_with_broker():
    live = broker_positions()
    if live != expected(basket):
        log_critical("desync"); adopt(live); set_basket_protection(); return False
    if any(p.sl is None for p in live):                        # SL missing => re-apply
        set_basket_protection()
    return True

on_new_day():  day_pl = 0 ; halted_day = False ; consec_losses = 0
```

---

## 12. Backtesting and validation methodology

### 12.1 Engine requirements

The engine must consume **bid and ask ticks**, never candle closes, because the strategy's exit target (0.6 × ATR) is frequently smaller than an M1 range and because grid triggers and stops can both be touched inside one bar.

Required model components: variable spread from the tick file · commission · swap (with triple-swap day) · slippage draw (log-normal, calibrated to observed latency) · broker lot step and minimum lot · margin and leverage · margin call and stop-out · execution delay (100–500 ms sampled) · order-rejection injection (0.5–2%) · intratick sequencing (SL checked before TP within a tick when both are inside the bid-ask range) · overnight and weekend gaps.

The engine **must not import MetaTrader5, must not open a socket to a broker, and must not place orders.** `research_backtest.py` (supplied) enforces this with a module-level guard.

### 12.2 Chronological splits

| Split | Purpose |
|---|---|
| First 60% | strategy discovery only |
| Next 20% | parameter selection only |
| Final 20% | **touched once**, out-of-sample |
| **+6–12 further months** | independent confirmation on data never used in any of the above |

**The 17 trading days available here are not sufficient for any of these splits.** A 60/20/20 split of 17 days gives 3.4 days of out-of-sample data — statistically meaningless. Data acquisition is therefore the first task, not optimisation.

This was verified empirically: `research_backtest.py` was executed end-to-end on the supplied tick file with a US$20,000 / 1:30 configuration. It ran cleanly and produced **0 / 0 / 1 trades** across the three splits — the ATR-percentile regime filter alone needs ~30 days of warm-up before it emits anything. The engine works; the data does not exist yet.

```
       split  trades  net    ret_pct  max_dd_pct  worst_trade
discovery_60       0    -          -           -            -
selection_20       0    -          -           -            -
      oos_20       1  -10.99   -0.05        0.00       -10.99
```

### 12.3 Required analyses before any demo deployment

Walk-forward (rolling 3-month train / 1-month test) · parameter-stability surfaces (performance must be a plateau, not a spike) · Monte Carlo trade-order resampling (≥20,000 paths) · Monte Carlo slippage and rejection · regime testing (trending / ranging / high-vol gold separately) · sensitivity to spread, ATR period and grid multiplier.

### 12.4 Reporting target

Report **P(reach the 6% Bootcamp Step-1 target before touching each drawdown level)**:

| Breach level | Required |
|---|---|
| 1% | report |
| 2% | report |
| 3% | report |
| 4% | report |
| **5% (Bootcamp hard limit)** | **P(breach) must be < 5%** before demo deployment |

This mirrors the hard gate already established in your EURUSD work (`P(maxDD > 4.5%)` treated as binary). Apply it identically here.

---

## 13. Final Go / No-Go

### Original bot: **NO-GO**

Five of the nine stated NO-GO conditions are met, each independently sufficient:

| NO-GO condition | Status |
|---|---|
| Depends on copied signals | ✅ **MET** — 2,721/2,721 deals are `copy #…` from an external master, which Bootcamp prohibits outright |
| Original entry signal cannot be independently generated | ✅ **MET** — best candidate rule: 2.61% precision vs 1.01% base rate |
| Losses cannot be bounded with visible stops | ✅ **MET** — 0 stop-losses on 2,721 orders; Bootcamp requires one on every position |
| Relies on uncapped Martingale | ✅ **MET** — 1.30 geometric ladder, no observed level cap, no loss-side exit |
| Cannot remain comfortably inside the maximum-loss rules | ✅ **MET** — tick-reconstructed floating drawdown **−7.67%** against a **5%** Bootcamp limit, in a month with **zero** losing days |
| Resembles prohibited tick scalping / HFT | ⚠️ **UNCLEAR** — 160 orders/day with sub-second bursts; requires written confirmation |
| Fails out-of-sample testing | ⚠️ **UNTESTABLE** — 17 days of data, three of them missing |
| Realistic costs remove profitability | ⚠️ **UNTESTABLE** — zero commission and a 50% credit bonus make the observed economics non-transferable |
| Tail risk unacceptable | ✅ **MET** — a 1% intraday adverse move already implies −5.8% of balance on the fitted 16-level ladder; 2% implies −20.7% |

There is no parameter setting that fixes this. The strategy's win rate *is* its risk: the 90.3% comes from refusing to realise losses.

### Path forward: **REDESIGN REQUIRED**

The independent strategy in §10 is a legitimate research candidate but **carries no validated performance claim**. Before it earns even a demo test it needs, in order:

1. **6–12 months of same-broker XAUUSD bid/ask tick data** plus a filled-in symbol specification. Nothing else can start until this exists.
2. The complete statement set, including 6 July, 13–14 July and the 17→20 July discontinuity, so the −3,210.88 USC of unobserved losses can be attributed.
3. A 60/20/20 chronological validation with `P(maxDD > 5%) < 5%` on Monte Carlo.
4. Written confirmation from The5ers on items #6 and #7 of the compliance matrix.

Only if all four clear does the verdict become **GO FOR FURTHER DEMO TESTING**. Today it does not.

---

## Appendix — files produced

| File | Contents |
|---|---|
| `XAUUSD_Bot_Reverse_Engineering_Report.md` | this report |
| `trades_reconstructed.csv` | 1,356 matched round-trip positions |
| `baskets_reconstructed.csv` / `baskets_with_mae.csv` | 486 baskets, with tick-derived MAE/MFE and floating loss |
| `raw_orders.csv` `raw_deals.csv` `raw_positions.csv` `raw_summary.csv` | unmodified statement sections with sheet provenance |
| `bursts.csv` · `entry_steps.csv` · `basket_ladder_fits.csv` · `basket_starts_features.csv` | intermediate model-fitting datasets |
| `charts_overview.png` | 12 charts (lot progression, basket depth, grid distance, entry gaps, duration, P/L, MAE, floating DD, balance, equity, time-of-day, stress) |
| `parse_statement.py` · `reconstruct.py` · `analyse_sizing_grid_exit.py` · `tick_analysis.py` · `entry_signal_and_charts.py` | the full research pipeline |
| `research_backtest.py` | tick-level research backtester for the §10 strategy (no broker connectivity) |
| `reconstruction_log.txt` · `analysis_log.txt` · `tick_analysis_log.txt` · `entry_signal_log.txt` | complete numeric audit trail |

**Sources for The5ers rules:** [Challenge Programs Explained (the5ers.com, updated 25 Jun 2026)](https://the5ers.com/challenge-programs-bootcamp-high-stakes-hyper-growth-explained/) · [Bootcamp programme page](https://the5ers.com/bootcamp/) · [Prohibited Trading Practices](https://the5ers.com/faqs/prohibited-trading-practices/)
