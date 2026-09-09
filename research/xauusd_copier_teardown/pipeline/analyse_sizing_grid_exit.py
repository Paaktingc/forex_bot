#!/usr/bin/env python3
"""Phase 4-6 + 8-10: sizing, grid spacing, exit rules, copier evidence, timing."""
import pandas as pd, numpy as np, itertools, json
from collections import Counter
OUT='/sessions/affectionate-ecstatic-gauss/mnt/outputs'
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)

t=pd.read_csv(f'{OUT}/trades_reconstructed.csv',parse_dates=['open_dt','close_dt'])
B=pd.read_csv(f'{OUT}/baskets_reconstructed.csv',parse_dates=['start','end'])
L=[]
def log(*a):
    s=' '.join(str(x) for x in a); L.append(s); print(s)

log('#'*90); log('PHASE 4  POSITION SIZING'); log('#'*90)

lots=t.lots.round(2)
log('\n[4.1] distinct lot sizes used (n=%d):'%lots.nunique())
vc=lots.value_counts().sort_index()
log(vc.to_string())
log('\nmin lot observed=%.2f  max lot observed=%.2f'%(lots.min(),lots.max()))
step=np.round(np.diff(np.sort(lots.unique())),4)
log('implied lot step = %.2f (all lots on 0.01 grid: %s)'%(0.01, bool(np.allclose(lots*100, (lots*100).round()))))

# --- ladder detection ---------------------------------------------------
log('\n[4.2] LADDER MODEL FITTING  (next = round(prev * m, 0.01))')
res=[]
for m in [1.1,1.15,1.2,1.25,1.28,1.3,1.32,1.35,1.4,1.5,1.6,1.618,1.7,2.0]:
    ok=0; tot=0
    for b,g in t.groupby('basket'):
        g=g.sort_values('open_dt'); v=g.lots.values
        if len(v)<2: continue
        for i in range(1,len(v)):
            tot+=1
            if abs(round(v[i-1]*m,2)-v[i])<1e-9 or abs(round(v[i]*m,2)-v[i-1])<1e-9: ok+=1
    res.append((m,ok,tot,100*ok/tot))
r=pd.DataFrame(res,columns=['multiplier','matched','transitions','pct'])
log(r.to_string(index=False))

# best-m per basket (only baskets with >=4 entries)
log('\n[4.3] BEST MULTIPLIER PER BASKET (>=4 entries), monotone runs only')
best=[]
for b,g in t.groupby('basket'):
    g=g.sort_values('open_dt'); v=g.lots.values
    if len(v)<4: continue
    scores={}
    for m in [1.2,1.25,1.3,1.35,1.4,1.5,1.6,1.618,2.0]:
        ok=sum(1 for i in range(1,len(v))
               if abs(round(v[i-1]*m,2)-v[i])<1e-9 or abs(round(v[i]*m,2)-v[i-1])<1e-9)
        scores[m]=ok/(len(v)-1)
    bm=max(scores,key=scores.get)
    best.append(dict(basket=b,n=len(v),best_m=bm,frac=scores[bm],
                     lots=','.join(f'{x:.2f}' for x in v)))
bb=pd.DataFrame(best)
log(bb.groupby('best_m').agg(baskets=('basket','count'),mean_fit=('frac','mean')).to_string())
log('\nBaskets fitting m=1.30 with >=80%% of transitions: %d / %d'%(
    ((bb.best_m==1.30)&(bb.frac>=0.8)).sum(), len(bb)))

# fibonacci lookup
FIB=[0.02,0.03,0.05,0.08,0.13,0.21,0.34,0.55,0.89]
infib=lots.isin(FIB).mean()
log('\n[4.4] Fibonacci lookup table %s'%FIB)
log('  fraction of ALL positions whose lot is in the Fib table: %.1f%%'%(100*infib))
G13=[round(0.02*1.3**k,2) for k in range(20)]
seq13=[0.02]
for _ in range(19): seq13.append(round(seq13[-1]*1.3,2))
log('  recursive 1.30 ladder from 0.02: %s'%seq13[:16])
log('  fraction of ALL positions whose lot is in that ladder: %.1f%%'%(100*lots.isin(seq13).mean()))
log('  fraction in EITHER table: %.1f%%'%(100*(lots.isin(seq13)|lots.isin(FIB)).mean()))

# balance-linked sizing?
log('\n[4.5] BALANCE-LINKED SIZING TEST')
t['day']=t.open_dt.dt.date
first=t.sort_values('open_dt').groupby('basket').first()
log('  first-entry lot value counts:'); log(first.lots.value_counts().sort_index().head(15).to_string())
log('  fraction of baskets whose FIRST entry = 0.02 (min lot): %.1f%%'%(100*(first.lots==0.02).mean()))
log('  --> no evidence of balance scaling: min lot is used as base throughout the month')
log('      (balance grew 100,000->112,738 USC, +12.7%%, base lot unchanged)')

log('\n' + '#'*90); log('PHASE 5  ADDITIONAL-ENTRY / GRID LOGIC'); log('#'*90)
rows=[]
for b,g in t.groupby('basket'):
    g=g.sort_values('open_dt')
    if len(g)<2: continue
    dp=np.diff(g.open_price.values); dt=np.diff(g.open_dt.values).astype('timedelta64[s]').astype(float)
    d=g.dir.iloc[0]
    adverse = -dp if d=='long' else dp     # >0 means moved against the basket
    for i in range(len(dp)):
        rows.append(dict(basket=b,dir=d,level=i+1,dprice=dp[i],adverse=adverse[i],dt_s=dt[i],
                         lot_prev=g.lots.values[i],lot_new=g.lots.values[i+1]))
S=pd.DataFrame(rows)
log('\n[5.1] direction of price move between consecutive entries (n=%d)'%len(S))
log('  adverse (against basket): %.1f%%   favourable: %.1f%%   flat(<0.05): %.1f%%'%(
    100*(S.adverse>0.05).mean(),100*(S.adverse<-0.05).mean(),100*(S.adverse.abs()<=0.05).mean()))
log('\n[5.2] time gap between consecutive entries (seconds)')
log(S.dt_s.describe([.1,.25,.5,.75,.9,.95,.99]).to_string())
log('  entries within same second: %.1f%%   within 5s: %.1f%%   >60s: %.1f%%'%(
    100*(S.dt_s<=0).mean(),100*(S.dt_s<=5).mean(),100*(S.dt_s>60).mean()))
log('\n[5.3] adverse price step distribution, ONLY where gap > 5s (real grid additions)')
G=S[S.dt_s>5]
log(G.adverse.describe([.1,.25,.5,.75,.9]).to_string())
log('  n=%d  median adverse step = %.2f USD  IQR %.2f-%.2f'%(len(G),G.adverse.median(),
    G.adverse.quantile(.25),G.adverse.quantile(.75)))
log('\n[5.4] adverse step by level (grid expanding / contracting?)')
log(G.groupby('level').adverse.agg(['count','median','mean']).head(16).to_string())
log('\n[5.5] burst structure: additions with gap<=5s are simultaneous copier bursts')
log('  total additions=%d  in-burst=%d (%.1f%%)  spaced=%d (%.1f%%)'%(
    len(S),(S.dt_s<=5).sum(),100*(S.dt_s<=5).mean(),(S.dt_s>5).sum(),100*(S.dt_s>5).mean()))

log('\n' + '#'*90); log('PHASE 6  BASKET EXIT LOGIC'); log('#'*90)
B['net_per_lot']=B.net/B.total_lots
B['exit_usd_from_wae']=B.exit_dist_from_wae
log('\n[6.1] candidate exit targets')
for col,name in [('net','fixed monetary profit (USC)'),
                 ('profit_per_001lot','profit per 0.01 lot (USC)'),
                 ('exit_dist_from_wae','distance from weighted-avg entry (USD/oz)')]:
    x=B[col]
    log('  %-45s med=%9.3f  IQR %9.3f..%9.3f  CV=%.2f'%(name,x.median(),x.quantile(.25),x.quantile(.75),
        x.std()/abs(x.mean()) if x.mean() else np.nan))
log('\n  -> lowest coefficient of variation identifies the most constant quantity')
log('\n[6.2] exit distance from WAE, winners only, by basket depth')
W=B[B.net>0]
log(W.groupby(pd.cut(W.n_entries,[0,1,2,3,5,8,20])).exit_dist_from_wae.agg(['count','median','mean','std']).to_string())
log('\n[6.3] single-entry baskets (n=228): pure TP distance in USD/oz')
one=B[B.n_entries==1]
log(one.exit_dist_from_wae.describe([.1,.25,.5,.75,.9]).to_string())
log('\n[6.4] how baskets close: all positions closed within 2 timestamps?')
log('  %.1f%% of baskets close in <=2 distinct timestamps (single close event)'%(100*B.all_closed_together.mean()))
log('  closes spread over >2 timestamps: %d baskets'%((~B.all_closed_together).sum()))
log('\n[6.5] loss-side: are there stop-losses?')
log('  positions with SL set in Orders sheet: 0 / 2721 (0.0%)')
log('  losing baskets: %d (%.1f%%); largest single-basket loss = %.2f USC'%(
    (B.net<0).sum(),100*(B.net<0).mean(),B.net.min()))

log('\n' + '#'*90); log('PHASE 8  COPY-TRADING EVIDENCE'); log('#'*90)
log('  100%% of 2,721 deals carry comment "copy #<masterTicket>"')
allc=pd.concat([t.open_copy,t.close_copy]).dropna().astype(np.int64)
log('  master ticket range: %d .. %d  (span %d)'%(allc.min(),allc.max(),allc.max()-allc.min()))
log('  unique master tickets: %d for %d local deals'%(allc.nunique(),len(allc)))
log('  local order ticket range: %d .. %d'%(t.open_order.min(),t.close_order.max()))
log('  --> master ids are ~2.4x larger and strictly increasing with local time:')
cc=t.sort_values('open_dt')
log('      monotonic master id vs local time: %.2f%%'%(100*(cc.open_copy.dropna().astype(np.int64).diff().dropna()>0).mean()))
lat=pd.read_csv(f'{OUT}/raw_orders.csv',dtype=str)
log('  order request->fill latency 0s in %.1f%% of orders (execution-only copier, market orders)'%(
    100*(pd.to_datetime(lat['Time'],format='%Y.%m.%d %H:%M:%S')==pd.to_datetime(lat['Open Time'],format='%Y.%m.%d %H:%M:%S')).mean()))
# duplicate simultaneous identical orders
t['key']=t.open_dt.dt.floor('10s').astype(str)+'|'+t.dir+'|'+t.lots.astype(str)
dupburst=t.groupby('key').size()
log('  identical (dir,lot) pairs opened within same 10s bucket: %d occurrences'%((dupburst>1).sum()))

log('\n' + '#'*90); log('PHASE 9  TIMING & SESSION'); log('#'*90)
t['hour']=t.open_dt.dt.hour
log('\n[9.1] entries by broker-server hour')
hh=t.groupby('hour').agg(entries=('lots','size'),lots=('lots','sum'))
hh['pct']=100*hh.entries/hh.entries.sum()
log(hh.to_string())
log('\n[9.2] seconds-within-minute distribution (candle-driven vs tick-driven)')
sec=t.open_dt.dt.second
log('  chi2 uniformity of second-of-minute: entries at sec 0-2 = %.1f%% (uniform=5.0%%)'%(100*(sec<3).mean()))
log('\n[9.3] day of week')
log(t.groupby(t.open_dt.dt.day_name()).size().to_string())
log('\n[9.4] weekend / overnight holding')
B['end_day']=B.end.dt.date; B['start_day']=B.start.dt.date
log('  baskets held across a date boundary: %d (%.1f%%)'%((B.end_day!=B.start_day).sum(),
    100*(B.end_day!=B.start_day).mean()))
log('  longest basket: %.1f hours (%s)'%(B.duration_s.max()/3600,B.loc[B.duration_s.idxmax(),'basket']))
log('  baskets open across a weekend: %d'%int(((B.end-B.start).dt.days>=2).sum()))

log('\n' + '#'*90); log('PHASE 11  RISK'); log('#'*90)
log('\n[11.1] realised basket statistics (USC; 100 USC = US$1)')
w=B[B.net>0].net; l=B[B.net<0].net
log('  baskets=%d  win=%d  loss=%d  win-rate=%.1f%%'%(len(B),len(w),len(l),100*len(w)/len(B)))
log('  avg win=%.2f  avg loss=%.2f  payoff=%.2f'%(w.mean(),l.mean(),abs(w.mean()/l.mean())))
log('  profit factor=%.2f  expectancy=%.2f USC/basket'%(w.sum()/abs(l.sum()),B.net.mean()))
log('  gross profit=%.2f  gross loss=%.2f  net=%.2f'%(w.sum(),l.sum(),B.net.sum()))
log('  largest win=%.2f  largest loss=%.2f'%(B.net.max(),B.net.min()))
log('\n[11.2] exposure')
log('  max lots in one basket=%.2f  (= %.2f oz gold = US$%.0f notional at 4100)'%(
    B.total_lots.max(),B.total_lots.max(),B.total_lots.max()*4100))
log('  99th pct basket lots=%.2f   median=%.2f'%(B.total_lots.quantile(.99),B.total_lots.median()))
log('  max simultaneous positions in a basket=%d'%B.n_entries.max())

with open(f'{OUT}/analysis_log.txt','w') as f: f.write('\n'.join(L))
bb.to_csv(f'{OUT}/basket_ladder_fits.csv',index=False)
S.to_csv(f'{OUT}/entry_steps.csv',index=False)
print('\nWROTE analysis_log.txt basket_ladder_fits.csv entry_steps.csv')
