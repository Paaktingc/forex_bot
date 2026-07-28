"""
Gate 3 — discovery-window signal test (2007-2018) for the frozen intraday
volatility-expansion continuation signal. Discovery data only; the locked OOS
period is never touched. Writes results/gate3_discovery.txt and appends the
test registry.

    python -m research.fx_intraday_volatility_continuation.run_gate3
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from . import backtest as B
from . import baselines as BL
from . import prepare_data as P
from . import signal as S
from .bootstrap import summary, iid_ci, block_ci

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
REGISTRY = os.path.join(HERE, "test_registry.csv")
SPLIT = "discovery"


def _pf(r):
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return g / l if l > 0 else float("inf")


def run():
    P.assert_research_only()
    lines = []
    def P_(s=""):
        lines.append(s); print(s)

    P_("=" * 92)
    P_(f"GATE 3 — discovery {C.DISCOVERY[0]}..{C.DISCOVERY[1]}  (frozen signal, after-cost verdicts)")
    P_("=" * 92)

    trades = {}
    pools = {}
    for pair in C.PAIRS:
        df = P.load_split(pair, SPLIT)
        sigs = S.accepted_signals(df)
        tr = B.run(df, pair, signals=sigs)
        trades[pair] = tr
        pools[pair] = BL.pool_frame(df, pair)
        r = tr["r"].to_numpy()
        st = summary(r)
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        drag = tr["r_beforecost"].mean() - r.mean()
        P_(f"\n{pair}: n={st['n']} ({st['n']/yrs:.0f}/yr)  E[R]net={st['mean']:+.4f} "
           f"gross={tr['r_beforecost'].mean():+.4f} costdrag={drag:.4f}")
        P_(f"  med={st['median']:+.3f} sd={st['sd']:.3f} WR={st['wr']:.3f} PF={st['pf']:.3f} "
           f"maxDD={st['maxdd_r']:.1f}R streak={st['longest_loss']}")
        P_(f"  iid95%[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] block95%[{st['bci_lo']:+.4f},{st['bci_hi']:+.4f}]")
        lr, sr = r[tr["direction"] == 1], r[tr["direction"] == -1]
        P_(f"  long n={len(lr)} E[R]={lr.mean():+.4f}  short n={len(sr)} E[R]={sr.mean():+.4f}")
        yr = tr.assign(y=[t.year for t in tr["entry_time"]]).groupby("y")["r"].mean()
        P_("  by year: " + " ".join(f"{y}:{m:+.2f}" for y, m in yr.items()))

    # ---- pooling ----
    allr = np.concatenate([trades[p]["r"].to_numpy() for p in C.PAIRS])
    noneu = np.concatenate([trades[p]["r"].to_numpy() for p in C.PAIRS if p != "EURUSD"])
    P_("\n" + "-" * 92)
    P_(f"POOLED 4-pair: n={len(allr)} E[R]={allr.mean():+.4f} PF={_pf(allr):.3f} "
       f"iid95%{tuple(round(x,4) for x in iid_ci(allr))}")
    P_(f"POOLED non-EURUSD: n={len(noneu)} E[R]={noneu.mean():+.4f} PF={_pf(noneu):.3f} "
       f"iid95%{tuple(round(x,4) for x in iid_ci(noneu))}")

    # leave-one-pair-out + remove-best-pair
    pair_sum = {p: trades[p]["r"].sum() for p in C.PAIRS}
    total = sum(pair_sum.values())
    P_("leave-one-out pooled E[R]:")
    for p in C.PAIRS:
        rest = np.concatenate([trades[q]["r"].to_numpy() for q in C.PAIRS if q != p])
        P_(f"  drop {p}: E[R]={rest.mean():+.4f} (n={len(rest)})")
    best_pair = max(pair_sum, key=lambda k: trades[k]["r"].mean())
    rest = np.concatenate([trades[q]["r"].to_numpy() for q in C.PAIRS if q != best_pair])
    P_(f"remove best pair ({best_pair}): pooled E[R]={rest.mean():+.4f}")

    # remove best year
    allyr = pd.Series(allr, index=np.concatenate(
        [[t.year for t in trades[p]["entry_time"]] for p in C.PAIRS]))
    yearmean = allyr.groupby(level=0).mean()
    best_year = yearmean.idxmax()
    P_(f"remove best year ({best_year}): pooled E[R]={allr[allyr.index != best_year].mean():+.4f}")

    # concentration of winners
    wins = np.sort(allr[allr > 0])[::-1]
    tot_win = wins.sum()
    for pct in (0.01, 0.05, 0.10):
        k = max(1, int(len(wins) * pct))
        P_(f"top {pct*100:.0f}% winners contribute {wins[:k].sum()/tot_win*100:.1f}% of gross wins "
           f"({k} trades)")

    # ---- baselines (pooled) ----
    P_("\n" + "-" * 92)
    P_("BASELINES (pooled, after-cost):")
    # matched random: per-pair sum arrays combined
    tot_n = sum(len(trades[p]) for p in C.PAIRS)
    mr_sum = np.zeros(C.N_BOOT)
    for p in C.PAIRS:
        st = trades[p].copy(); st["year"] = [t.year for t in st["entry_time"]]
        mr = BL.matched_random(st, pools[p])   # mean per pair
        mr_sum += mr * len(trades[p])
    mr_pooled = mr_sum / tot_n
    strat_mean = allr.mean()
    p_ge = (mr_pooled >= strat_mean).mean()
    P_(f"  matched-random: mean={mr_pooled.mean():+.4f} 95%[{np.percentile(mr_pooled,2.5):+.4f},"
       f"{np.percentile(mr_pooled,97.5):+.4f}]  P(random>=strategy)={p_ge:.3f}  "
       f"strategy pct-rank={(mr_pooled<strat_mean).mean()*100:.1f}")

    # shuffled labels: pooled over signal bars
    sh_sum = np.zeros(C.N_BOOT)
    for p in C.PAIRS:
        sh = BL.shuffled_labels(trades[p].set_index("signal_time"), pools[p])
        sh_sum += sh * len(trades[p])
    sh_pooled = sh_sum / tot_n
    P_(f"  shuffled-label: mean={sh_pooled.mean():+.4f} 95%[{np.percentile(sh_pooled,2.5):+.4f},"
       f"{np.percentile(sh_pooled,97.5):+.4f}]  P(shuffle>=strategy)={(sh_pooled>=strat_mean).mean():.3f}")

    # unconditional session continuation
    unc = np.concatenate([BL.unconditional_continuation(P.load_split(p, SPLIT), p) for p in C.PAIRS])
    P_(f"  unconditional session continuation: n={len(unc)} E[R]={unc.mean():+.4f} "
       f"(strategy {strat_mean:+.4f}; improvement {strat_mean-unc.mean():+.4f})")

    # frequency bucket
    yrs_all = np.mean([(P.load_split(p, SPLIT).index[-1] - P.load_split(p, SPLIT).index[0]).days/365.25
                       for p in C.PAIRS])
    tpy = tot_n / yrs_all
    bucket = ("<150" if tpy < 150 else "150-300" if tpy < 300 else
              "300-600" if tpy < 600 else ">600")
    P_(f"\nportfolio frequency ~{tpy:.0f} trades/year  bucket={bucket}")

    # ---- verdict ----
    P_("\n" + "=" * 92)
    fails = []
    if allr.mean() <= 0: fails.append("pooled after-cost E[R] non-positive")
    if noneu.mean() <= 0: fails.append("pooled non-EURUSD E[R] negative")
    if (mr_pooled >= strat_mean).mean() > 0.05: fails.append("does not beat matched-random at 95%")
    if (sh_pooled >= strat_mean).mean() > 0.05: fails.append("does not beat shuffled-label at 95%")
    verdict = "GATE 3 FAILED" if fails else "GATE 3 PASSED"
    P_(f"{verdict}. " + ("; ".join(fails) if fails else "all criteria met"))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "gate3_discovery.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    _registry_append([
        dict(timestamp=datetime.now(timezone.utc).isoformat(), config_id="gate3_frozen_v1",
             split=SPLIT, params="ATR20,exp1.5,loc0.75,bo8,stop1.0ATR,tgt1.5R,time16,cd4h,sess08-16L",
             reason="Gate 3 discovery test of frozen signal",
             result="results/gate3_discovery.txt",
             pooled_after_cost_ER=round(float(allr.mean()), 4), verdict=verdict),
    ])
    return verdict


def _registry_append(rows):
    cols = ["timestamp", "config_id", "split", "params", "reason", "result",
            "pooled_after_cost_ER", "verdict"]
    new = not os.path.exists(REGISTRY)
    with open(REGISTRY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    run()
