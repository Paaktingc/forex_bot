#!/usr/bin/env python3
"""
Phase 2-3: Reconstruct transaction-level and basket-level datasets.

Key calibrated facts (derived, see validate step):
  * XAUUSD-STDc on a USC (US-cent) account
  * P/L(cents) = (exit-entry) * lots * 100   for LONG
  * P/L(cents) = (entry-exit) * lots * 100   for SHORT
    => 1 lot = 1 troy ounce, P/L reported in US cents. 100 USC = US$1.
"""
import re, math, json
import pandas as pd, numpy as np

OUT = '/sessions/affectionate-ecstatic-gauss/mnt/outputs'
CENTS_PER_DOLLAR_PER_LOT = 100.0   # calibrated multiplier

def load():
    d = pd.read_csv(f'{OUT}/raw_deals.csv', dtype=str)
    o = pd.read_csv(f'{OUT}/raw_orders.csv', dtype=str)
    return d, o

def n(s):
    if pd.isna(s): return np.nan
    s = str(s).replace('\xa0',' ').replace(' ','').replace(',','')
    if s in ('','-'): return np.nan
    try: return float(s)
    except: return np.nan

def main():
    d, o = load()
    rep = []
    def log(x):
        rep.append(x); print(x)

    log("="*78)
    log("PHASE 1/2  RECONSTRUCTION LOG")
    log("="*78)

    # ---- balance / credit operations -------------------------------------
    bal = d[d['Type'].isin(['balance','credit'])].copy()
    log("\n[non-trade operations]")
    log(bal[['Open Time','Ticket','Type','Profit','Comment','_sheet']].to_string(index=False))

    d = d[d['Type'].isin(['buy','sell'])].copy()

    # ---- dedupe ----------------------------------------------------------
    before = len(d)
    dupmask = d.duplicated(subset=['Ticket'], keep='first')
    dups = d[d['Ticket'].duplicated(keep=False)]
    log(f"\n[dedupe] deal rows before={before}  duplicate-ticket rows={len(dups)}  "
        f"dropped={dupmask.sum()}  after={before-dupmask.sum()}")
    log(f"[dedupe] duplicated tickets appear in sheets: "
        f"{sorted(dups['_sheet'].unique().tolist())}")
    # verify duplicates are byte-identical on the trade fields
    chk = dups.groupby('Ticket')[['Open Time','Type','Size','Price','Entry','Profit','Swap']].nunique()
    log(f"[dedupe] duplicated tickets with conflicting field values: "
        f"{int((chk>1).any(axis=1).sum())}  -> {'CLEAN (exact repeats)' if int((chk>1).any(axis=1).sum())==0 else 'CONFLICT'}")
    d = d[~dupmask].copy()

    # ---- typed columns ---------------------------------------------------
    d['dt']    = pd.to_datetime(d['Open Time'], format='%Y.%m.%d %H:%M:%S')
    d['deal']  = d['Ticket'].astype(np.int64)
    d['order'] = d['Order'].astype(np.int64)
    d['lots']  = d['Size'].map(n)
    d['price'] = d['Price'].map(n)
    d['swap']  = d['Swap'].map(n).fillna(0.0)
    d['comm']  = d['Commission'].map(n).fillna(0.0)
    d['fee']   = d['Fee'].map(n).fillna(0.0)
    d['profit']= d['Profit'].map(n).fillna(0.0)
    d['side']  = d['Type']          # deal side
    d['entry'] = d['Entry']
    d = d.sort_values(['dt','deal']).reset_index(drop=True)

    # copy id from comment
    d['copy_id'] = d['Comment'].str.extract(r'copy #(\d+)')[0]

    log(f"\n[deals] n={len(d)}  span {d['dt'].min()} .. {d['dt'].max()}")
    log(f"[deals] in={(d.entry=='in').sum()}  out={(d.entry=='out').sum()}")
    log(f"[deals] commission total={d['comm'].sum()}  fee total={d['fee'].sum()}  swap total={d['swap'].sum():.2f}")
    log(f"[deals] copy-comment coverage: {d['copy_id'].notna().sum()}/{len(d)} "
        f"({100*d['copy_id'].notna().mean():.2f}%)")

    # ---- orders ----------------------------------------------------------
    o = o[~o['Ticket'].duplicated(keep='first')].copy()
    o['order'] = o['Ticket'].astype(np.int64)
    o['req_dt']  = pd.to_datetime(o['Open Time'], format='%Y.%m.%d %H:%M:%S')
    o['exec_dt'] = pd.to_datetime(o['Time'],      format='%Y.%m.%d %H:%M:%S')
    o['lat_s']   = (o['exec_dt']-o['req_dt']).dt.total_seconds()
    o['req_lots'] = o['Size'].str.split('/').str[0].map(n)
    o['fil_lots'] = o['Size'].str.split('/').str[1].map(n)
    log(f"\n[orders] n={len(o)}  all state='filled': {(o['State']=='filled').all()}")
    log(f"[orders] partial fills (req!=filled): {(o.req_lots!=o.fil_lots).sum()}")
    log(f"[orders] all SL=0: {(o['S / L']=='0').all()}   all TP=0: {(o['T / P']=='0').all()}")
    log(f"[orders] all price='market': {(o['Price']=='market').all()}")
    log(f"[orders] request->fill latency (s): "
        f"min={o.lat_s.min():.0f} med={o.lat_s.median():.0f} p95={o.lat_s.quantile(.95):.0f} max={o.lat_s.max():.0f}")

    d = d.merge(o[['order','req_dt','exec_dt','lat_s','req_lots','fil_lots']], on='order', how='left')
    log(f"[join] deals without matching order row: {d['req_dt'].isna().sum()}")

    # =====================================================================
    # POSITION MATCHING
    # =====================================================================
    # implied entry price for every 'out' deal, from reported profit
    def implied_entry(r):
        if r.entry != 'out' or r.lots == 0: return np.nan
        delta = r.profit / (r.lots * CENTS_PER_DOLLAR_PER_LOT)
        # out deal side 'sell' closes a LONG:  profit=(exit-entry)*L*100
        # out deal side 'buy'  closes a SHORT: profit=(entry-exit)*L*100
        return r.price - delta if r.side == 'sell' else r.price + delta
    d['impl_entry'] = d.apply(implied_entry, axis=1)

    open_pos = []      # list of dicts
    trades   = []      # completed round-trips
    unmatched_out = []
    pid = 0
    for r in d.itertuples():
        if r.entry == 'in':
            pid += 1
            open_pos.append(dict(pos_id=r.order, seq=pid, dir=('long' if r.side=='buy' else 'short'),
                                 open_dt=r.dt, open_price=r.price, lots=r.lots,
                                 open_deal=r.deal, open_order=r.order, open_copy=r.copy_id,
                                 open_lat=r.lat_s, orig_lots=r.lots))
        else:
            want_dir = 'short' if r.side=='buy' else 'long'
            remaining = r.lots
            # candidate set: same direction, still open
            cands = [p for p in open_pos if p['dir']==want_dir]
            # rank: exact lot match & price match first, then FIFO
            def score(p):
                pm = abs(p['open_price'] - r.impl_entry) if not math.isnan(r.impl_entry) else 9e9
                lm = abs(p['lots'] - r.lots)
                return (0 if (pm<0.01 and lm<1e-9) else (1 if pm<0.01 else 2), lm, pm, p['seq'])
            cands.sort(key=score)
            for p in cands:
                if remaining <= 1e-9: break
                take = min(remaining, p['lots'])
                frac = take / r.lots
                trades.append(dict(
                    pos_id=p['pos_id'], dir=p['dir'],
                    open_dt=p['open_dt'], open_price=p['open_price'], lots=take,
                    close_dt=r.dt, close_price=r.price,
                    implied_entry=r.impl_entry,
                    price_match_err=abs(p['open_price']-r.impl_entry) if not math.isnan(r.impl_entry) else np.nan,
                    partial_open=(take < p['orig_lots']-1e-9),
                    partial_close=(take < r.lots-1e-9),
                    profit=r.profit*frac, swap=r.swap*frac,
                    comm=r.comm*frac, fee=r.fee*frac,
                    open_deal=p['open_deal'], close_deal=r.deal,
                    open_order=p['open_order'], close_order=r.order,
                    open_copy=p['open_copy'], close_copy=r.copy_id,
                    open_lat=p['open_lat'], close_lat=r.lat_s))
                p['lots'] -= take
                remaining -= take
            open_pos = [p for p in open_pos if p['lots'] > 1e-9]
            if remaining > 1e-9:
                unmatched_out.append(dict(deal=r.deal, dt=r.dt, side=r.side,
                                          lots=r.lots, unmatched=remaining, price=r.price))

    t = pd.DataFrame(trades).sort_values(['open_dt','pos_id']).reset_index(drop=True)
    log("\n" + "="*78)
    log("POSITION MATCHING")
    log("="*78)
    log(f"[match] round-trip trades reconstructed : {len(t)}")
    log(f"[match] positions still open at end     : {len(open_pos)}  "
        f"(lots={sum(p['lots'] for p in open_pos):.2f})")
    log(f"[match] unmatched 'out' volume events   : {len(unmatched_out)}")
    if unmatched_out:
        log(pd.DataFrame(unmatched_out).to_string(index=False))
    log(f"[match] exact price match (<0.01)       : "
        f"{(t.price_match_err<0.01).sum()}/{len(t)} ({100*(t.price_match_err<0.01).mean():.2f}%)")
    log(f"[match] median |implied-actual| entry   : {t.price_match_err.median():.5f}")
    log(f"[match] worst  |implied-actual| entry   : {t.price_match_err.max():.5f}")
    log(f"[match] partial opens={t.partial_open.sum()}  partial closes={t.partial_close.sum()}")

    # reconciliation vs statement closed P/L
    log(f"\n[recon] sum reconstructed profit = {t.profit.sum():,.2f} USC")
    log(f"[recon] sum reconstructed swap   = {t.swap.sum():,.2f} USC")
    log(f"[recon] sum profit+swap          = {(t.profit+t.swap).sum():,.2f} USC")
    log(f"[recon] sum of all 'out' deal profit+swap in deals table = "
        f"{(d.loc[d.entry=='out','profit'].sum()+d.loc[d.entry=='out','swap'].sum()):,.2f} USC")

    t['duration_s'] = (t.close_dt-t.open_dt).dt.total_seconds()
    t['net']  = t.profit + t.swap + t.comm + t.fee
    t['pips'] = np.where(t.dir=='long', t.close_price-t.open_price, t.open_price-t.close_price)
    t.to_csv(f'{OUT}/trades_reconstructed.csv', index=False)

    # =====================================================================
    # BASKET GROUPING
    # =====================================================================
    # A basket = maximal run of same-direction positions during which that
    # direction's open exposure never returns to zero.
    log("\n" + "="*78)
    log("BASKET GROUPING")
    log("="*78)
    ev = []
    for r in t.itertuples():
        ev.append((r.open_dt, 1, r.dir, r.Index))
        ev.append((r.close_dt, -1, r.dir, r.Index))
    ev.sort(key=lambda x: (x[0], x[1]))       # closes before opens at same ts
    bid = {'long':0,'short':0}; expo={'long':0,'short':0}; cur={'long':None,'short':None}
    assign = {}
    for ts, s, dr, ix in ev:
        if s == 1:
            if expo[dr] == 0:
                bid[dr] += 1; cur[dr] = f"{dr[0].upper()}{bid[dr]:03d}"
            expo[dr] += 1
            assign[ix] = cur[dr]
        else:
            expo[dr] -= 1
    t['basket'] = t.index.map(assign)

    rows = []
    for b, g in t.groupby('basket'):
        g = g.sort_values('open_dt')
        lots = g.lots.values; ep = g.open_price.values; xp = g.close_price.values
        wae = float((ep*lots).sum()/lots.sum())
        wax = float((xp*lots).sum()/lots.sum())
        # max simultaneous exposure
        e2 = sorted([(r.open_dt,r.lots) for r in g.itertuples()] +
                    [(r.close_dt,-r.lots) for r in g.itertuples()], key=lambda x:(x[0], x[1]))
        run=0; mx=0
        for _,v in e2:
            run+=v; mx=max(mx,run)
        gaps = g.open_dt.diff().dt.total_seconds().dropna()
        pdist= np.diff(ep)
        rows.append(dict(
            basket=b, dir=g.dir.iloc[0],
            start=g.open_dt.min(), end=g.close_dt.max(),
            duration_s=(g.close_dt.max()-g.open_dt.min()).total_seconds(),
            n_entries=len(g), total_lots=float(lots.sum()), max_simul_lots=float(mx),
            wavg_entry=wae, wavg_exit=wax,
            first_entry=float(ep[0]), last_entry=float(ep[-1]),
            entry_span=float(ep.max()-ep.min()),
            gross_profit=float(g.profit.sum()), swap=float(g.swap.sum()),
            comm=float(g.comm.sum()), net=float(g.net.sum()),
            profit_per_001lot=float(g.net.sum()/(lots.sum()/0.01)),
            exit_dist_from_wae=float(wax-wae if g.dir.iloc[0]=='long' else wae-wax),
            n_close_events=int(g.close_dt.nunique()),
            all_closed_together=bool(g.close_dt.nunique()<=2),
            med_gap_s=float(gaps.median()) if len(gaps) else np.nan,
            max_gap_s=float(gaps.max()) if len(gaps) else np.nan,
            med_price_step=float(np.median(np.abs(pdist))) if len(pdist) else np.nan,
            n_partial=int(g.partial_open.sum()+g.partial_close.sum()),
        ))
    B = pd.DataFrame(rows).sort_values('start').reset_index(drop=True)
    B['outcome'] = np.where(B.net>0,'profit', np.where(B.net<0,'loss','flat'))
    B.to_csv(f'{OUT}/baskets_reconstructed.csv', index=False)
    t.to_csv(f'{OUT}/trades_reconstructed.csv', index=False)   # re-save with basket ids

    log(f"[baskets] total = {len(B)}   long={(B.dir=='long').sum()}  short={(B.dir=='short').sum()}")
    log(f"[baskets] entries per basket: min={B.n_entries.min()} med={B.n_entries.median():.0f} "
        f"p90={B.n_entries.quantile(.9):.0f} max={B.n_entries.max()}")
    log(f"[baskets] net P/L: winners={ (B.net>0).sum() }  losers={ (B.net<0).sum() }  "
        f"win-rate={100*(B.net>0).mean():.1f}%")
    log(f"[baskets] total net = {B.net.sum():,.2f} USC")
    log(f"[baskets] duration s: med={B.duration_s.median():.0f} p90={B.duration_s.quantile(.9):.0f} max={B.duration_s.max():.0f}")
    log(f"[baskets] max total lots in one basket = {B.total_lots.max():.2f} "
        f"(max simultaneous {B.max_simul_lots.max():.2f})")

    # hedging check
    log("\n[hedging] periods with simultaneous long+short exposure:")
    tl = t.copy()
    marks=[]
    for r in tl.itertuples():
        marks.append((r.open_dt, 1, r.dir)); marks.append((r.close_dt, -1, r.dir))
    marks.sort(key=lambda x:(x[0],x[1]))
    L=S=0; hedged=0; tot=0; last=None; hedged_time=0.0
    for ts,s,dr in marks:
        if last is not None and L>0 and S>0:
            hedged_time += (ts-last).total_seconds()
        if dr=='long': L+=s
        else: S+=s
        last=ts
    log(f"  seconds with both directions open = {hedged_time:,.0f}s "
        f"({hedged_time/3600:.1f}h) -> genuine hedging {'PRESENT' if hedged_time>0 else 'ABSENT'}")

    with open(f'{OUT}/reconstruction_log.txt','w') as f:
        f.write('\n'.join(str(x) for x in rep))
    log("\nWrote trades_reconstructed.csv, baskets_reconstructed.csv, reconstruction_log.txt")

if __name__ == '__main__':
    main()
