#!/usr/bin/env python3
"""
research_backtest.py — tick-level RESEARCH backtester for the independent,
risk-capped XAUUSD basket strategy specified in section 10 of the report.

    *** THIS CODE MUST NEVER TRADE. ***
It has no broker connectivity by design and refuses to run if MetaTrader5 or
any broker SDK is importable in the process. It reads CSV files and writes CSV
files. Nothing else.

Usage
-----
    python3 research_backtest.py --ticks xauusd-tick.csv \
        --balance 20000 --contract 100 --leverage 30 \
        --split 0.6,0.2,0.2 --out results/

Tick CSV must have columns: timestamp (epoch ms, UTC), bidPrice, askPrice.
"""
from __future__ import annotations
import argparse, sys, os, json, math
import numpy as np, pandas as pd

# ───────────────────────── SAFETY GUARD ─────────────────────────
_FORBIDDEN = ('MetaTrader5', 'mt5', 'ctrader_open_api', 'ib_insync', 'ccxt')
for _m in _FORBIDDEN:
    if _m in sys.modules:
        raise SystemExit(f"REFUSING TO RUN: broker module '{_m}' is loaded. "
                         "This is research-only code and must not touch a live account.")


# ───────────────────────── CONFIG ─────────────────────────
class Cfg:
    # risk
    risk_pct        = 0.0025      # per basket, of STARTING balance
    daily_stop_pct  = 0.0075
    total_stop_pct  = 0.0200
    # ladder
    max_additions   = 3
    ladder          = (1.00, 1.30, 1.60, 1.90)
    # distances (in ATR units)
    stop_atr        = 2.0
    tp_atr          = 0.6
    grid_atr        = 0.8
    # filters
    max_spread      = 1.20        # USD/oz
    spread_mult     = 2.0         # vs trailing 60-min median
    atr_lo_pct, atr_hi_pct = 20, 90
    session         = (7, 19)     # server hours, UTC+3
    max_duration_s  = 4 * 3600
    cooldown_s      = 3600
    max_consec_loss = 2
    # instrument
    contract        = 100.0       # oz per lot (standard)
    lot_step        = 0.01
    min_lot         = 0.01
    leverage        = 30.0
    # costs
    commission_per_lot_rt = 7.0   # USD, round turn — set from broker
    swap_long_per_lot_day = -15.0
    swap_short_per_lot_day = 5.0
    slippage_mu, slippage_sd = 0.02, 0.03   # USD/oz, log-normal-ish
    exec_delay_ms   = (100, 500)
    reject_prob     = 0.01
    server_tz_offset_h = 3


# ───────────────────────── DATA ─────────────────────────
def load_ticks(path: str) -> pd.DataFrame:
    tk = pd.read_csv(path, dtype={'timestamp': np.int64,
                                  'bidPrice': np.float64, 'askPrice': np.float64})
    assert tk.timestamp.is_monotonic_increasing, "ticks not sorted"
    assert (tk.askPrice > tk.bidPrice).all(), "crossed spread in source data"
    tk['ts'] = (pd.to_datetime(tk.timestamp, unit='ms', utc=True)
                  .dt.tz_convert(None) + pd.Timedelta(hours=Cfg.server_tz_offset_h))
    return tk


def build_bars(tk: pd.DataFrame, rule: str = '15min') -> pd.DataFrame:
    mid = (tk.askPrice + tk.bidPrice) / 2
    d = pd.DataFrame({'ts': tk.ts, 'mid': mid, 'sp': tk.askPrice - tk.bidPrice})
    b = d.set_index('ts').resample(rule).agg(
        o=('mid', 'first'), h=('mid', 'max'), l=('mid', 'min'), c=('mid', 'last'),
        spread=('sp', 'median'), n=('mid', 'size')).dropna()
    return b


def add_indicators(b: pd.DataFrame) -> pd.DataFrame:
    c = b.c
    b['ema200'] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    rs = (d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() /
          (-d).clip(lower=0).ewm(alpha=1/14, adjust=False).mean())
    b['rsi14'] = 100 - 100 / (1 + rs)
    tr = pd.concat([b.h - b.l, (b.h - c.shift()).abs(), (b.l - c.shift()).abs()], axis=1).max(axis=1)
    b['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    b['atr_h1'] = b['atr'] * 2.0            # crude H1 proxy from M15; replace with true H1
    m = c.rolling(20).mean(); s = c.rolling(20).std()
    b['bb_up'], b['bb_dn'] = m + 2*s, m - 2*s
    b['atr_lo'] = b.atr.rolling(2880, min_periods=200).quantile(Cfg.atr_lo_pct/100)
    b['atr_hi'] = b.atr.rolling(2880, min_periods=200).quantile(Cfg.atr_hi_pct/100)
    b['sp_med60'] = b.spread.rolling(4, min_periods=1).median()
    return b


# ───────────────────────── STRATEGY ─────────────────────────
def signal(row, prev) -> str | None:
    if not (Cfg.session[0] <= row.name.hour < Cfg.session[1]):        return None
    if row.spread > Cfg.max_spread:                                    return None
    if row.spread > Cfg.spread_mult * row.sp_med60:                    return None
    if not (row.atr_lo <= row.atr <= row.atr_hi):                      return None
    if abs(row.c - row.ema200) > 1.5 * row.atr_h1:                     return None
    if prev.c < prev.bb_dn and row.c > row.bb_dn and row.rsi14 < 30:   return 'long'
    if prev.c > prev.bb_up and row.c < row.bb_up and row.rsi14 > 70:   return 'short'
    return None


def size_basket(balance0, atr_h1):
    """Return (base_lot, n_rungs, stop_dist) or None if it cannot be sized safely."""
    risk = balance0 * Cfg.risk_pct
    stop_d = Cfg.stop_atr * atr_h1
    if stop_d <= 0: return None
    for n in range(Cfg.max_additions + 1, 0, -1):
        w = sum(Cfg.ladder[:n])
        raw = risk / (w * Cfg.contract * stop_d)
        lot = math.floor(raw / Cfg.lot_step) * Cfg.lot_step
        if lot < Cfg.min_lot:
            continue
        worst = w * lot * Cfg.contract * stop_d
        if worst <= risk * 1.05:
            return round(lot, 2), n, stop_d
    return None


class Basket:
    __slots__ = ('dir','lots','prices','open_ts','stop_d','n_rungs','degraded')
    def __init__(self, d, lot, px, ts, stop_d, n):
        self.dir, self.lots, self.prices = d, [lot], [px]
        self.open_ts, self.stop_d, self.n_rungs, self.degraded = ts, stop_d, n, False
    @property
    def wae(self):
        L = sum(self.lots)
        return sum(p*l for p, l in zip(self.prices, self.lots)) / L
    @property
    def total_lots(self): return sum(self.lots)
    def sl(self):
        return self.wae - self.stop_d if self.dir == 'long' else self.wae + self.stop_d
    def tp(self, atr):
        k = Cfg.tp_atr * atr
        return self.wae + k if self.dir == 'long' else self.wae - k


def run(bars: pd.DataFrame, ticks: pd.DataFrame, balance0: float, rng) -> pd.DataFrame:
    """Bar-driven decisions, tick-driven fills and stop/target checks."""
    tarr = ticks.ts.values.astype('datetime64[ns]')
    bid, ask = ticks.bidPrice.values, ticks.askPrice.values

    balance = balance0; peak = balance0
    day = None; day_pl = 0.0; halted_day = False; halted_total = False
    cooldown_until = pd.Timestamp.min; consec = 0
    basket: Basket | None = None
    trades = []

    def slip(rng): return max(0.0, rng.normal(Cfg.slippage_mu, Cfg.slippage_sd))

    def close_basket(ts, px, reason):
        nonlocal basket, balance, day_pl, consec, cooldown_until, halted_day
        b = basket
        gross = sum((px - p) * l if b.dir == 'long' else (p - px) * l
                    for p, l in zip(b.prices, b.lots)) * Cfg.contract
        comm = Cfg.commission_per_lot_rt * b.total_lots
        days = max(0, (ts - b.open_ts).days)
        sw = days * b.total_lots * (Cfg.swap_long_per_lot_day if b.dir == 'long'
                                    else Cfg.swap_short_per_lot_day)
        net = gross - comm + sw
        balance += net; day_pl += net
        trades.append(dict(open_ts=b.open_ts, close_ts=ts, dir=b.dir,
                           n=len(b.lots), lots=b.total_lots, wae=b.wae,
                           exit=px, gross=gross, comm=comm, swap=sw, net=net,
                           reason=reason, balance=balance))
        consec = consec + 1 if net < 0 else 0
        if net < 0: cooldown_until = ts + pd.Timedelta(seconds=Cfg.cooldown_s)
        if consec >= Cfg.max_consec_loss: halted_day = True
        basket = None

    prev = None
    for ts, row in bars.iterrows():
        if prev is None or row[['atr','ema200','bb_dn','atr_lo']].isna().any():
            prev = row; continue
        if ts.date() != day:
            day, day_pl, halted_day, consec = ts.date(), 0.0, False, 0

        # ---- walk ticks inside this bar for stop / target / grid ----
        i0 = np.searchsorted(tarr, np.datetime64(ts))
        i1 = np.searchsorted(tarr, np.datetime64(ts + pd.Timedelta(bars.index.freq or '15min')))
        if basket is not None:
            for i in range(i0, i1):
                b = basket
                px_close = bid[i] if b.dir == 'long' else ask[i]
                sl, tp = b.sl(), b.tp(row.atr_h1)
                # intratick sequencing: stop is checked BEFORE target (conservative)
                if (b.dir == 'long' and px_close <= sl) or (b.dir == 'short' and px_close >= sl):
                    close_basket(pd.Timestamp(tarr[i]), sl - slip(rng) if b.dir=='long' else sl + slip(rng), 'stop'); break
                if (b.dir == 'long' and px_close >= tp) or (b.dir == 'short' and px_close <= tp):
                    close_basket(pd.Timestamp(tarr[i]), tp, 'target'); break
                # grid addition
                if len(b.lots) < b.n_rungs and not b.degraded:
                    adverse = (b.prices[-1] - px_close) if b.dir == 'long' else (px_close - b.prices[-1])
                    if adverse >= Cfg.grid_atr * row.atr_h1 and (ask[i]-bid[i]) <= Cfg.max_spread:
                        if rng.random() < Cfg.reject_prob:
                            b.degraded = True                    # failsafe: no retry
                        else:
                            lot = round(b.lots[0] * Cfg.ladder[len(b.lots)], 2)
                            fill = (ask[i] + slip(rng)) if b.dir == 'long' else (bid[i] - slip(rng))
                            b.lots.append(lot); b.prices.append(fill)
                # timeout
                if (pd.Timestamp(tarr[i]) - b.open_ts).total_seconds() > Cfg.max_duration_s:
                    close_basket(pd.Timestamp(tarr[i]), px_close, 'timeout'); break
                # drawdown monitors
                fl = sum((px_close - p) * l if b.dir=='long' else (p - px_close) * l
                         for p, l in zip(b.prices, b.lots)) * Cfg.contract
                if day_pl + fl <= -Cfg.daily_stop_pct * balance0:
                    close_basket(pd.Timestamp(tarr[i]), px_close, 'daily_stop'); halted_day = True; break
                if (balance + fl - balance0) <= -Cfg.total_stop_pct * balance0:
                    close_basket(pd.Timestamp(tarr[i]), px_close, 'total_stop'); halted_total = True; break

        peak = max(peak, balance)
        if halted_total: break

        # ---- new basket decision on bar close ----
        if basket is None and not halted_day and ts >= cooldown_until:
            d = signal(row, prev)
            if d:
                sized = size_basket(balance0, row.atr_h1)
                if sized:
                    lot, n, stop_d = sized
                    j = min(i1, len(tarr) - 1)
                    fill = (ask[j] + slip(rng)) if d == 'long' else (bid[j] - slip(rng))
                    # margin check
                    if sum(Cfg.ladder[:n]) * lot * Cfg.contract * fill / Cfg.leverage < 0.25 * balance:
                        basket = Basket(d, lot, fill, ts, stop_d, n)
        prev = row

    return pd.DataFrame(trades)


def report(tr: pd.DataFrame, balance0: float, label: str) -> dict:
    if tr.empty: return dict(split=label, trades=0)
    eq = balance0 + tr.net.cumsum()
    dd = (eq - eq.cummax()) / eq.cummax()
    w, l = tr[tr.net > 0].net, tr[tr.net < 0].net
    return dict(split=label, trades=len(tr),
                win_rate=round(100 * len(w) / len(tr), 1),
                pf=round(w.sum() / abs(l.sum()), 2) if len(l) else np.inf,
                net=round(tr.net.sum(), 2),
                ret_pct=round(100 * tr.net.sum() / balance0, 2),
                max_dd_pct=round(100 * dd.min(), 2),
                worst_trade=round(tr.net.min(), 2),
                stops=int((tr.reason == 'stop').sum()),
                targets=int((tr.reason == 'target').sum()),
                daily_stops=int((tr.reason == 'daily_stop').sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticks', required=True)
    ap.add_argument('--balance', type=float, default=20000)
    ap.add_argument('--contract', type=float, default=100)
    ap.add_argument('--leverage', type=float, default=30)
    ap.add_argument('--split', default='0.6,0.2,0.2')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    Cfg.contract, Cfg.leverage = a.contract, a.leverage
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    tk = load_ticks(a.ticks)
    bars = add_indicators(build_bars(tk))
    fr = [float(x) for x in a.split.split(',')]
    n = len(bars); cuts = [0, int(n*fr[0]), int(n*(fr[0]+fr[1])), n]
    names = ['discovery_60', 'selection_20', 'oos_20']

    rows = []
    for k in range(3):
        sub = bars.iloc[cuts[k]:cuts[k+1]]
        if len(sub) < 300:
            rows.append(dict(split=names[k], trades=0, note='insufficient data')); continue
        sl = tk[(tk.ts >= sub.index[0]) & (tk.ts <= sub.index[-1])]
        tr = run(sub, sl, a.balance, rng)
        tr.to_csv(f'{a.out}/trades_{names[k]}.csv', index=False)
        rows.append(report(tr, a.balance, names[k]))

    R = pd.DataFrame(rows)
    R.to_csv(f'{a.out}/summary.csv', index=False)
    print(R.to_string(index=False))
    print("\nWARNING: results on <6 months of data are not evidence of an edge.")
    print("Required gate before any demo deployment: P(maxDD > 5%) < 5% "
          "over >=20,000 Monte Carlo trade-order resamples on >=6 months of OOS data.")


if __name__ == '__main__':
    main()
