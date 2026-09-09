#!/usr/bin/env python3
"""Phase 1 (tz) + 3 (MAE/MFE) + 7 (entry signal) + 11 (floating DD) using Dukascopy ticks."""
import pandas as pd, numpy as np, os
OUT='/sessions/affectionate-ecstatic-gauss/mnt/outputs'
TICK='/sessions/affectionate-ecstatic-gauss/mnt/uploads/xauusd-tick-2026-07-01-2026-07-26T23-33.csv'
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
L=[]
def log(*a):
    s=' '.join(str(x) for x in a); L.append(s); print(s)

log('#'*90); log('TICK DATA VALIDATION (Dukascopy XAUUSD, epoch-ms UTC)'); log('#'*90)
tk=pd.read_csv(TICK, dtype={'timestamp':np.int64,'askPrice':np.float64,'bidPrice':np.float64})
log('rows=%d'%len(tk))
tk['ts']=pd.to_datetime(tk.timestamp,unit='ms',utc=True)
log('span UTC: %s .. %s'%(tk.ts.min(),tk.ts.max()))
log('monotonic timestamps: %s'%tk.timestamp.is_monotonic_increasing)
log('duplicate timestamps: %d'%tk.timestamp.duplicated().sum())
log('null bid/ask: %d / %d'%(tk.bidPrice.isna().sum(),tk.askPrice.isna().sum()))
tk['spread']=tk.askPrice-tk.bidPrice
log('crossed/zero spread rows (ask<=bid): %d'%(tk.spread<=0).sum())
log('spread USD: med=%.3f p90=%.3f p99=%.3f max=%.3f'%(tk.spread.median(),tk.spread.quantile(.9),
    tk.spread.quantile(.99),tk.spread.max()))
log('price range: bid %.2f .. %.2f'%(tk.bidPrice.min(),tk.bidPrice.max()))
gap=tk.ts.diff().dt.total_seconds()
log('tick gaps >300s: %d   >3600s: %d  (weekends)'%((gap>300).sum(),(gap>3600).sum()))
big=tk.loc[gap>3600,'ts']
log('gaps >1h start times (UTC):');
for a,b in zip(tk.ts[gap>3600].tolist(), gap[gap>3600].tolist()):
    log('   ends %s  after %.1f h'%(a, b/3600))

# ---------- TIMEZONE CALIBRATION -------------------------------------------
log('\n'+'#'*90); log('BROKER SERVER TIMEZONE CALIBRATION'); log('#'*90)
t=pd.read_csv(f'{OUT}/trades_reconstructed.csv',parse_dates=['open_dt','close_dt'])
t=t[t.open_dt>='2026-07-01']
tkv=tk[['timestamp','bidPrice','askPrice']].values
ts_ms=tk.timestamp.values
bid=tk.bidPrice.values; ask=tk.askPrice.values
def err_for_offset(off_h):
    """mean |statement price - dukascopy mid| when statement time is shifted by off_h."""
    e=[]
    sub=t.sample(min(600,len(t)),random_state=1)
    for r in sub.itertuples():
        u=(r.open_dt - pd.Timedelta(hours=off_h)).tz_localize('UTC')
        i=np.searchsorted(ts_ms,int(u.value//10**6))
        if i<=0 or i>=len(ts_ms): continue
        mid=(bid[i]+ask[i])/2
        e.append(abs(r.open_price-mid))
    return np.median(e), len(e)
log('\noffset(h)  median|stmt price - duka mid|  n')
best=None
for off in range(-6,9):
    m,n=err_for_offset(off)
    flag=''
    if best is None or m<best[1]: best=(off,m)
    log('  %+3d        %10.3f              %d'%(off,m,n))
log('\nBEST OFFSET = server time is UTC%+d  (median price error %.3f USD)'%(best[0],best[1]))
OFF=best[0]
log('=> statement timestamps are UTC%+d ; UK (BST, UTC+1) = server %+d h'%(OFF, 1-OFF))

# ---------- MAE / MFE / FLOATING P/L ---------------------------------------
log('\n'+'#'*90); log('BASKET MAE / MFE / FLOATING DRAWDOWN (tick-reconstructed)'); log('#'*90)
t_all=pd.read_csv(f'{OUT}/trades_reconstructed.csv',parse_dates=['open_dt','close_dt'])
t_all['open_utc']=t_all.open_dt-pd.Timedelta(hours=OFF)
t_all['close_utc']=t_all.close_dt-pd.Timedelta(hours=OFF)
# resample ticks to 1s mid for speed
tk['sec']=tk.timestamp//1000
sec=tk.groupby('sec').agg(bid=('bidPrice','last'),ask=('askPrice','last'),
                          hi=('askPrice','max'),lo=('bidPrice','min')).reset_index()
sarr=sec['sec'].values; shi=sec.hi.values; slo=sec.lo.values; sbid=sec.bid.values; sask=sec.ask.values
log('1-second bars built: %d'%len(sec))

B=pd.read_csv(f'{OUT}/baskets_reconstructed.csv',parse_dates=['start','end'])
res=[]
CUT=pd.Timestamp('2026-07-01')
for b,g in t_all.groupby('basket'):
    if g.open_dt.min()<CUT:            # no tick coverage before 1 July
        res.append(dict(basket=b,mae_usd=np.nan,mfe_usd=np.nan,min_float=np.nan,max_float=np.nan,covered=False)); continue
    s0=int(g.open_utc.min().value//10**9); s1=int(g.close_utc.max().value//10**9)
    i0=np.searchsorted(sarr,s0); i1=np.searchsorted(sarr,s1)+1
    if i1<=i0:
        res.append(dict(basket=b,mae_usd=np.nan,mfe_usd=np.nan,min_float=np.nan,max_float=np.nan,covered=False)); continue
    tsec=sarr[i0:i1]; hi=shi[i0:i1]; lo=slo[i0:i1]
    # floating P/L in USC at each second, only positions already opened
    opens=np.array([int(x.value//10**9) for x in g.open_utc])
    lots=g.lots.values; ep=g.open_price.values
    d=g.dir.iloc[0]
    live=(tsec[:,None]>=opens[None,:])
    if d=='long':
        worst=(lo[:,None]-ep[None,:])*lots[None,:]*100
        bestv=(hi[:,None]-ep[None,:])*lots[None,:]*100
    else:
        worst=(ep[None,:]-hi[:,None])*lots[None,:]*100
        bestv=(ep[None,:]-lo[:,None])*lots[None,:]*100
    fw=np.where(live,worst,0).sum(axis=1); fb=np.where(live,bestv,0).sum(axis=1)
    res.append(dict(basket=b,mae_usd=float(fw.min()/ (g.lots.sum()*100)),
                    mfe_usd=float(fb.max()/(g.lots.sum()*100)),
                    min_float=float(fw.min()), max_float=float(fb.max()), covered=True))
R=pd.DataFrame(res)
B=B.merge(R,on='basket',how='left')
B.to_csv(f'{OUT}/baskets_with_mae.csv',index=False)
C=B[B.covered==True]
log('baskets with tick coverage: %d / %d'%(len(C),len(B)))
log('\nmax floating LOSS per basket (USC, negative):')
log(C.min_float.describe([.5,.75,.9,.95,.99]).to_string())
log('\nWORST 10 baskets by floating loss:')
log(C.nsmallest(10,'min_float')[['basket','dir','start','n_entries','total_lots','min_float','net','duration_s']].to_string(index=False))
log('\nrealised net vs floating loss:')
log('  median (floating loss / realised net) for winners = %.2f'%
    (C[C.net>0].min_float.abs()/C[C.net>0].net).median())
log('  worst basket floating loss = %.2f USC = US$%.2f = %.2f%% of peak equity (162,738 USC)'%
    (C.min_float.min(), C.min_float.min()/100, 100*C.min_float.min()/162738))

# ---------- ACCOUNT-LEVEL FLOATING EQUITY ----------------------------------
log('\n'+'#'*90); log('ACCOUNT-LEVEL FLOATING EQUITY (all open baskets simultaneously)'); log('#'*90)
tt=t_all[t_all.open_dt>=CUT].copy()
grid=sarr[(sarr>=int(tt.open_utc.min().value//10**9))&(sarr<=int(tt.close_utc.max().value//10**9))]
gi0=np.searchsorted(sarr,grid[0])
ghi=shi[gi0:gi0+len(grid)]; glo=slo[gi0:gi0+len(grid)]
float_pl=np.zeros(len(grid))
worst_pl=np.zeros(len(grid))
for r in tt.itertuples():
    a=np.searchsorted(grid,int(r.open_utc.value//10**9)); z=np.searchsorted(grid,int(r.close_utc.value//10**9))
    if z<=a: continue
    if r.dir=='long': w=(glo[a:z]-r.open_price)*r.lots*100
    else:             w=(r.open_price-ghi[a:z])*r.lots*100
    worst_pl[a:z]+=w
log('worst simultaneous floating P/L across whole period = %.2f USC (US$%.2f)'%(worst_pl.min(),worst_pl.min()/100))
bal0=101942.72   # balance at 2026-07-01 open
log('as %% of balance at 1 July (101,942.72 USC): %.2f%%'%(100*worst_pl.min()/bal0))
log('as %% of equity incl. 50,000 credit (151,942.72): %.2f%%'%(100*worst_pl.min()/(bal0+50000)))
np.save(f'{OUT}/worst_floating.npy',worst_pl)
np.save(f'{OUT}/worst_floating_grid.npy',grid)

with open(f'{OUT}/tick_analysis_log.txt','w') as f: f.write('\n'.join(L))
print('\nDONE')
