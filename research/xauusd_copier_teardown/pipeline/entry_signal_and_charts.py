#!/usr/bin/env python3
"""Phase 7 (first-entry signal, precision/recall) + Phase 12 charts."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT='/sessions/affectionate-ecstatic-gauss/mnt/outputs'
TICK='/sessions/affectionate-ecstatic-gauss/mnt/uploads/xauusd-tick-2026-07-01-2026-07-26T23-33.csv'
OFF=3  # server = UTC+3
pd.set_option('display.width',300)
L=[]
def log(*a):
    s=' '.join(str(x) for x in a); L.append(s); print(s)

tk=pd.read_csv(TICK,dtype={'timestamp':np.int64,'askPrice':np.float64,'bidPrice':np.float64})
tk['ts']=pd.to_datetime(tk.timestamp,unit='ms',utc=True).dt.tz_convert(None)+pd.Timedelta(hours=OFF)
tk['mid']=(tk.askPrice+tk.bidPrice)/2
tk['sp']=tk.askPrice-tk.bidPrice
m1=tk.set_index('ts').resample('1min').agg(
    o=('mid','first'),h=('mid','max'),l=('mid','min'),c=('mid','last'),
    spread=('sp','mean'),n=('mid','size')).dropna()
log('M1 bars built from ticks (server time UTC+3): %d  %s .. %s'%(len(m1),m1.index[0],m1.index[-1]))

c=m1.c
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
m1['ema20']=ema(c,20); m1['ema50']=ema(c,50); m1['ema200']=ema(c,200)
d=c.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean()
m1['rsi14']=100-100/(1+rs)
tr=pd.concat([m1.h-m1.l,(m1.h-c.shift()).abs(),(m1.l-c.shift()).abs()],axis=1).max(axis=1)
m1['atr14']=tr.ewm(alpha=1/14,adjust=False).mean()
m1['bb_mid']=c.rolling(20).mean(); sd=c.rolling(20).std()
m1['bb_up']=m1.bb_mid+2*sd; m1['bb_dn']=m1.bb_mid-2*sd
m1['bbpos']=(c-m1.bb_dn)/(m1.bb_up-m1.bb_dn)
m1['ret5']=c.pct_change(5)*10000   # bp
m1['ret15']=c.pct_change(15)*10000
m1['hh20']=m1.h.rolling(20).max(); m1['ll20']=m1.l.rolling(20).min()
m1['slope20']=m1.ema20.diff(5)

t=pd.read_csv(f'{OUT}/trades_reconstructed.csv',parse_dates=['open_dt','close_dt'])
B=pd.read_csv(f'{OUT}/baskets_with_mae.csv',parse_dates=['start','end'])
starts=B[B.start>='2026-07-01 04:00'].copy()
starts['bar']=starts.start.dt.floor('1min')
X=m1.reindex(starts.bar.values)
starts=starts.reset_index(drop=True)
for col in ['rsi14','bbpos','atr14','ret5','ret15','slope20','spread','c','ema20','ema50','ema200','hh20','ll20']:
    starts[col]=X[col].values
starts=starts.dropna(subset=['rsi14'])
log('basket starts with full indicator coverage: %d (long=%d short=%d)'%(
    len(starts),(starts.dir=='long').sum(),(starts.dir=='short').sum()))

log('\n'+'#'*90); log('PHASE 7  FIRST-ENTRY SIGNAL ANALYSIS'); log('#'*90)
log('\n[7.1] market state at basket start, by direction (median)')
cols=['rsi14','bbpos','ret5','ret15','slope20','atr14','spread']
log(starts.groupby('dir')[cols].median().to_string())
log('\n[7.2] population baseline (all M1 bars in trading hours)')
base=m1[(m1.index.hour>=4)&(m1.index.hour<=23)].dropna()
log(base[cols].median().to_frame('all_bars').T.to_string())

log('\n[7.3] Is direction predicted by state? (long vs short separation)')
for col in cols:
    a=starts.loc[starts.dir=='long',col]; b=starts.loc[starts.dir=='short',col]
    from scipy import stats
    try: u,p=stats.mannwhitneyu(a,b)
    except Exception: p=np.nan
    log('  %-9s long_med=%9.3f  short_med=%9.3f   MWU p=%.4f %s'%(col,a.median(),b.median(),p,
        '<-- significant' if p<0.01 else ''))

log('\n[7.4] CANDIDATE SIGNAL PRECISION / RECALL')
log('    A signal is evaluated on all M1 bars 04:00-23:00 server; a bar is a TRUE POSITIVE')
log('    if a new basket of the matching direction started in that minute.')
truth={}
for r in starts.itertuples(): truth[(r.bar,r.dir)]=True
cand=base.copy()
def evaluate(name,mask_long,mask_short):
    tpL=sum(1 for b in cand.index[mask_long] if (b,'long') in truth)
    tpS=sum(1 for b in cand.index[mask_short] if (b,'short') in truth)
    fired=mask_long.sum()+mask_short.sum()
    tp=tpL+tpS
    actual=len(starts)
    prec=100*tp/fired if fired else 0
    rec=100*tp/actual
    log('  %-42s fired=%6d  TP=%3d  precision=%5.2f%%  recall=%5.1f%%'%(name,fired,tp,prec,rec))
    return prec,rec
evaluate('RSI<30 long / RSI>70 short (mean-rev)', (cand.rsi14<30).values, (cand.rsi14>70).values)
evaluate('RSI<35 long / RSI>65 short',            (cand.rsi14<35).values, (cand.rsi14>65).values)
evaluate('BB lower break long / upper short',     (cand.bbpos<0).values,  (cand.bbpos>1).values)
evaluate('trend: c>ema200 long / c<ema200 short', (cand.c>cand.ema200).values,(cand.c<cand.ema200).values)
evaluate('breakout: c>=hh20 long / c<=ll20 short',(cand.c>=cand.hh20).values,(cand.c<=cand.ll20).values)
evaluate('fade 5m move >8bp (counter-trend)',     (cand.ret5<-8).values,  (cand.ret5>8).values)
evaluate('momentum 5m move >8bp (with-trend)',    (cand.ret5>8).values,   (cand.ret5<-8).values)
evaluate('random baseline (every bar, both dirs)',np.ones(len(cand),bool),np.ones(len(cand),bool))

log('\n[7.5] restart-delay analysis (is entry timer-driven?)')
ss=B.sort_values('start')
gap=(ss.start-ss.end.shift()).dt.total_seconds()
log(gap.describe([.1,.25,.5,.75,.9]).to_string())
log('  new basket starts within 60s of previous basket close: %.1f%%'%(100*(gap<60).mean()))
log('  overlapping baskets (start before previous end): %.1f%%'%(100*(gap<0).mean()))

log('\n[7.6] CONCLUSION')
log('  No tested indicator rule reaches useful precision. Basket starts are spread')
log('  across the whole indicator distribution and are near-indistinguishable from')
log('  the unconditional bar population. The first-entry trigger is NOT recoverable')
log('  from this account: it lives on the master account. See Phase 8.')

# ============================= CHARTS =================================
log('\nBuilding charts...')
plt.rcParams.update({'figure.dpi':110,'font.size':8})
tt=t.copy()
Bc=pd.read_csv(f'{OUT}/baskets_with_mae.csv',parse_dates=['start','end'])
fig,ax=plt.subplots(4,3,figsize=(16,13))
A=ax.ravel()
# 1 lot progression
lv=[]
for b,g in tt.groupby('basket'):
    g=g.sort_values('open_dt')
    for i,v in enumerate(g.lots.values): lv.append((i,v))
LV=pd.DataFrame(lv,columns=['level','lot'])
md=LV.groupby('level').lot.median()
A[0].semilogy(md.index,md.values,'o-',label='observed median')
ref=[0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.11,0.14,0.18,0.23,0.30,0.39,0.51,0.67,0.87]
A[0].semilogy(range(len(ref)),ref,'r--',label='fitted ladder')
A[0].set_title('1. Lot progression by basket level'); A[0].set_xlabel('level'); A[0].legend()
# 2 basket depth
A[1].hist(Bc.n_entries,bins=range(1,18),color='steelblue',edgecolor='w')
A[1].set_title('2. Basket depth (entries per basket)')
# 3 grid distance
S=pd.read_csv(f'{OUT}/entry_steps.csv')
A[2].hist(S[S.dt_s>5].adverse.clip(-5,12),bins=60,color='darkorange',edgecolor='w')
A[2].axvline(S[S.dt_s>5].adverse.median(),color='k',ls='--')
A[2].set_title('3. Adverse grid step (USD/oz), gaps>5s\nmedian=%.2f'%S[S.dt_s>5].adverse.median())
# 4 time between entries
A[3].hist(np.log10(S.dt_s.clip(1)+1),bins=60,color='seagreen',edgecolor='w')
A[3].set_title('4. log10(seconds between entries)')
# 5 basket duration
A[4].hist(np.log10(Bc.duration_s.clip(1)),bins=50,color='purple',edgecolor='w')
A[4].set_title('5. log10(basket duration, s)')
# 6 profit per basket
A[5].plot(Bc.sort_values('start').net.values,'.',ms=3)
A[5].axhline(0,color='k',lw=.5); A[5].set_title('6. Net P/L per basket (USC)')
A[5].set_yscale('symlog')
# 7 MAE
Cc=Bc[Bc.covered==True]
A[6].scatter(Cc.total_lots,-Cc.min_float,s=8,alpha=.6)
A[6].set_xscale('log'); A[6].set_yscale('log')
A[6].set_title('7. Max floating loss vs basket lots'); A[6].set_xlabel('total lots'); A[6].set_ylabel('MAE (USC)')
# 8 floating DD timeline
wf=np.load(f'{OUT}/worst_floating.npy'); gd=np.load(f'{OUT}/worst_floating_grid.npy')
gt=pd.to_datetime(gd,unit='s')
A[7].fill_between(gt,wf,0,color='crimson',alpha=.6)
A[7].set_title('8. Account worst-case floating P/L (USC)')
A[7].tick_params(axis='x',rotation=30)
# 9 cumulative balance
tt2=tt.sort_values('close_dt')
A[8].plot(tt2.close_dt,100000+ (tt2.profit+tt2.swap).cumsum())
A[8].set_title('9. Reconstructed balance curve (USC)'); A[8].tick_params(axis='x',rotation=30)
# 10 equity curve (balance + floating)
_b=pd.Series((tt2.profit+tt2.swap).cumsum().values,index=tt2.close_dt)
_b=_b.groupby(level=0).last()
bal=_b.reindex(_b.index.union(gt)).ffill().reindex(gt).fillna(0)+100000
A[9].plot(gt,bal.values,label='balance')
A[9].plot(gt,bal.values+wf,label='equity (worst-case)',lw=.7)
A[9].legend(); A[9].set_title('10. Balance vs worst-case equity'); A[9].tick_params(axis='x',rotation=30)
# 11 time of day
hh=tt.groupby(tt.open_dt.dt.hour).size()
A[10].bar(hh.index,hh.values,color='teal'); A[10].set_title('11. Entries by server hour (UTC+3)')
# 12 stress test
lots=np.array([0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.11,0.14,0.18,0.23,0.30,0.39,0.51,0.67,0.87])
gridd=2.1
for nlv in [8,12,16]:
    ll=lots[:nlv]; dist=np.arange(nlv)*gridd
    moves=np.linspace(0,140,200)
    dd=[]
    for mv in moves:
        filled=dist<=mv
        dd.append(((mv-dist[filled])*ll[filled]).sum()*100)
    A[11].plot(moves,dd,label=f'max {nlv} levels')
A[11].axhline(101942*0.05,color='r',ls='--',label='5% of balance')
A[11].set_yscale('log'); A[11].legend()
A[11].set_title('12. Stress: floating loss (USC) vs adverse move (USD/oz)')
A[11].set_xlabel('adverse move USD/oz')
plt.tight_layout(); plt.savefig(f'{OUT}/charts_overview.png',dpi=130)
log('saved charts_overview.png')

with open(f'{OUT}/entry_signal_log.txt','w') as f: f.write('\n'.join(L))
starts.to_csv(f'{OUT}/basket_starts_features.csv',index=False)
print('DONE')
