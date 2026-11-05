#!/usr/bin/env python3
"""Generate a provisional, shareable research bundle on UCLA lobster2.

Reads licensed message archives in place. Do NOT commit raw/derived market data
without checking the lab agreement. Requires reconstruct_ofi.py in cwd,
NumPy, pandas, matplotlib, 7z. All quote reconstructions are PROVISIONAL.
"""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reconstruct_ofi import Book, ofi, OPEN, CLOSE

NBUCKET = (CLOSE - OPEN) // 300
METRICS = ('ofi_raw', 'ofi_per_volume', 'ofi_per_depth', 'executed_volume', 'mean_depth')


def one_archive(archive):
    book = Book()
    # raw OFI, executed volume proxy, depth-seconds, valid seconds, transitions
    acc = np.zeros((NBUCKET, 5), dtype=float)
    t_last = float(OPEN)
    previous, quote = None, None
    stats = {'rows': 0, 'bad_lines': 0, 'nonmonotonic': 0, 'invalid_quotes': 0}
    prev_t = -1.0
    first_regular_anchor = None

    def advance(t):
        nonlocal t_last
        t = min(max(t, OPEN), CLOSE)
        while t > t_last + 1e-10:
            j = min(int((t_last-OPEN)//300), NBUCKET-1)
            end = min(t, OPEN+300*(j+1))
            if quote is not None:
                acc[j, 2] += (quote[0][1]+quote[1][1])*(end-t_last)
                acc[j, 3] += end-t_last
            t_last = end

    proc = subprocess.Popen(['7z','x','-so',str(archive)], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        for row in csv.reader(proc.stdout):
            stats['rows'] += 1
            try:
                t=float(row[0]); typ=int(row[1]); oid=int(row[2]); qty=int(row[3]);
                price=int(row[4]); side=int(row[5])
            except (ValueError, IndexError):
                stats['bad_lines'] += 1
                continue
            if t < prev_t:
                stats['nonmonotonic'] += 1
            prev_t=max(prev_t, t)
            if t >= OPEN:
                advance(t)
            changed=book.update(typ, oid, qty, price, side)
            if t < OPEN:
                if changed:
                    quote=book.quote()
                continue
            if t >= CLOSE:
                continue
            j=int((t-OPEN)//300)
            if typ in (4,5) and qty>0:
                acc[j,1] += qty  # explicit executed-volume proxy; excludes crosses
            if changed:
                cur=book.quote()
                if cur is None:
                    stats['invalid_quotes'] += 1
                    previous=None
                    quote=None
                else:
                    if first_regular_anchor is None:
                        first_regular_anchor=t
                    if previous is not None:
                        acc[j,0] += ofi(previous,cur)
                        acc[j,4] += 1
                    previous=cur
                    quote=cur
        advance(CLOSE)
    finally:
        proc.stdout.close()
        rc=proc.wait()
    if rc:
        raise RuntimeError(f'7z exit code {rc} for {archive}')

    valid=acc[:,3]>=300-1e-5
    raw=np.where(valid,acc[:,0],np.nan)
    vol=acc[:,1]
    depth=np.where(valid,acc[:,2]/300,np.nan)
    ratio_vol=np.divide(raw,vol,out=np.full(NBUCKET,np.nan),where=vol>0)
    ratio_depth=np.divide(raw,depth,out=np.full(NBUCKET,np.nan),where=depth>0)
    stats.update({'valid_depth_buckets':int(np.isfinite(depth).sum()),
                  'zero_volume_buckets':int((vol==0).sum()),
                  'anchor':first_regular_anchor,
                  'remaining_orders':len(book.orders),
                  'book_warnings':dict(book.stats)})
    return dict(zip(METRICS,(raw,ratio_vol,ratio_depth,vol,depth))), stats


def build(args):
    universe=pd.read_csv(args.universe)
    tickers=universe.ticker.astype(str).tolist()[:args.limit or None]
    if len(tickers)!=len(set(tickers)): raise ValueError('Repeated ticker names')
    root=Path(args.data_root)
    days=args.dates or sorted(p.name for p in root.iterdir()
                              if p.is_dir() and len(p.name)==8 and p.name.isdigit())
    days=days[:args.max_days or None]
    if not days: raise ValueError('No trading days selected')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    mats={m:np.full((len(days),NBUCKET,len(tickers)),np.nan) for m in METRICS}
    qc=[]
    for d,day in enumerate(days):
        for j,ticker in enumerate(tickers):
            print(f'[{d+1}/{len(days)} day; {j+1}/{len(tickers)} stock] {day} {ticker}',flush=True)
            archive=root/day/(ticker+'.7z')
            if not archive.is_file():
                qc.append({'day':day,'ticker':ticker,'status':'missing_archive'})
                continue
            try:
                result,stats=one_archive(archive)
                for name in METRICS: mats[name][d,:,j]=result[name]
                qc.append({'day':day,'ticker':ticker,'status':'processed_provisional',
                           **{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in stats.items()}})
            except Exception as exc:
                qc.append({'day':day,'ticker':ticker,'status':'failed','error':str(exc)})
    clocks=[]
    for day in days:
        date=datetime.strptime(day,'%Y%m%d')
        clocks.extend([(date+timedelta(seconds=OPEN+300*i)).strftime('%Y-%m-%d %H:%M')
                       for i in range(NBUCKET)])
    for name,array in mats.items():
        wide=pd.DataFrame(array.reshape(-1,len(tickers)),columns=tickers)
        wide.insert(0,'minute',clocks)
        wide.to_csv(out/(name+'_5min_stacked.csv'),index=False)
        np.save(out/(name+'_tensor.npy'),array)
    # Historical clock-bucket average, strictly previous days. Renormalize for NaNs.
    hist=np.full_like(mats['ofi_raw'],np.nan)
    for d in range(1,len(days)):
        old=mats['ofi_raw'][:d]
        age=np.arange(d,0,-1)
        weights=2.0**(-age/args.half_life_days)
        valid=np.isfinite(old)
        numerator=(np.where(valid,old,0)*weights[:,None,None]).sum(axis=0)
        denominator=(valid*weights[:,None,None]).sum(axis=0)
        hist[d]=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    panel=pd.DataFrame(hist.reshape(-1,len(tickers)),columns=tickers)
    panel.insert(0,'minute',clocks)
    panel.to_csv(out/'historical_weighted_raw_ofi.csv',index=False)
    pd.DataFrame(qc).to_csv(out/'reconstruction_qc.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'days':days,'tickers':tickers,
      'tensor_axes':['day','five_minute_bucket','ticker'],
      'volume':'sum of type 4 and 5 execution share quantities; excludes type 6 cross trades',
      'depth':'time-weighted mean bid+ask displayed shares at level 1',
      'caution':'NO independent order-book snapshot validation; provisional'},indent=2))
    print('Saved daily panels and tensors:',out,'shape:',mats['ofi_raw'].shape)


def soft(x,t):return np.sign(x)*np.maximum(np.abs(x)-t,0)


def rpca(A,steps=150,tolerance=1e-5):
    # Inexact augmented Lagrangian Principal Component Pursuit: A=L+S+E
    lam=1/np.sqrt(max(A.shape))
    norm2=np.linalg.norm(A,2)
    Y=A/max(norm2,np.max(np.abs(A))/lam,1e-12)
    mu=1.25/max(norm2,1e-12)
    L=np.zeros_like(A);S=np.zeros_like(A)
    for it in range(steps):
        u,s,vt=np.linalg.svd(A-S+Y/mu,full_matrices=False)
        L=(u*np.maximum(s-1/mu,0))@vt
        S=soft(A-L+Y/mu,lam/mu)
        E=A-L-S
        err=np.linalg.norm(E)/(np.linalg.norm(A)+1e-12)
        if err<tolerance:break
        Y+=mu*E
        mu*=1.5
    return L,S,A-L-S,it+1,err


def factors(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    input_dir=Path(args.input)
    data=pd.read_csv(input_dir/(args.metric+'_5min_stacked.csv'))
    stocks=data.columns[1:].tolist()
    dates=data.minute.str[:10]
    distinct=sorted(dates.unique())
    if len(distinct)<2:raise ValueError('At least two trading days are required')
    train_days=args.train_days or len(distinct)-1
    if not 1<=train_days<len(distinct): raise ValueError('Invalid train-days')
    X=data[stocks].to_numpy(dtype=float)
    valid=np.isfinite(X).all(axis=1)
    train=dates.isin(distinct[:train_days]).to_numpy() & valid
    test=dates.isin(distinct[train_days:]).to_numpy() & valid
    if train.sum()<5 or test.sum()<2:raise ValueError('Too few complete train/test rows')
    mu=X[train].mean(axis=0); sigma=X[train].std(axis=0,ddof=1)
    if np.any(~np.isfinite(sigma)) or np.any(sigma<=0):raise ValueError('Invalid training standard deviations')
    Ztrain=(X[train]-mu)/sigma; Ztest=(X[test]-mu)/sigma
    weights=np.ones(train.sum())
    if args.half_life_days:
        age=(pd.to_datetime(dates[train]).max()-pd.to_datetime(dates[train])).dt.days.to_numpy()
        weights=np.power(0.5,age/args.half_life_days)
    _,s,vt=np.linalg.svd(Ztrain*np.sqrt(weights[:,None]),full_matrices=False)
    fraction=s*s/(s*s).sum(); cumulative=np.cumsum(fraction)
    threshold_rank=int(np.searchsorted(cumulative,args.variance_target)+1)
    yy=s/s[0]; chord=yy[0]+(yy[-1]-yy[0])*np.linspace(0,1,len(s))
    elbow=int(np.argmax(chord-yy)+1)
    k=args.rank or threshold_rank
    basis=vt[:k,:]
    low=(Ztest@basis.T)@basis
    residual=Ztest-low
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    test_time=data.loc[test,'minute'].to_numpy()
    def save(name,arr):
        frame=pd.DataFrame(arr,columns=stocks);frame.insert(0,'minute',test_time)
        frame.to_csv(out/(name+'.csv'),index=False)
    save('heldout_standardized',Ztest)
    save('svd_lowrank',low)
    save('svd_residual',residual)
    pd.DataFrame({'rank':np.arange(1,len(s)+1),'singular':s,'explained':fraction,
                  'cumulative':cumulative}).to_csv(out/'svd_rank_selection.csv',index=False)
    pd.DataFrame(basis.T,index=stocks,columns=[f'PC{i+1}' for i in range(k)]).to_csv(out/'svd_loadings.csv')
    # Fit robust low rank + sparse on TRAIN ONLY; project test onto learned space.
    L,S,E,steps,err=rpca(Ztrain,steps=args.rpca_steps)
    u,ss,v=np.linalg.svd(L,full_matrices=False)
    r=max(1,int((ss>1e-6*max(ss[0],1e-12)).sum()))
    rb=v[:r]
    low_rp=(Ztest@rb.T)@rb
    remainder=Ztest-low_rp
    # Threshold is trained on past only; held-out sparse split is heuristic.
    threshold=np.quantile(np.abs(S),0.95)
    if threshold<=0: threshold=np.quantile(np.abs(Ztrain-L),0.95)
    sparse=soft(remainder,threshold)
    noise=remainder-sparse
    save('rpca_lowrank',low_rp);save('rpca_sparse',sparse);save('rpca_error',noise)
    diag=[]
    df=pd.DataFrame(residual,columns=stocks)
    df['day']=pd.Series(test_time).str[:10]
    for lag in (1,3,6):
        for ticker in stocks:
            past=df.groupby('day')[ticker].shift(lag)
            future=df.groupby('day')[ticker].shift(-1)
            rolling=df.groupby('day')[ticker].transform(lambda x:x.rolling(lag,min_periods=lag).sum())
            diag.append({'ticker':ticker,'lookback_buckets':lag,
              'lag_corr':df[ticker].corr(past),
              'past_sum_to_next_corr':rolling.corr(future)})
    pd.DataFrame(diag).to_csv(out/'heldout_residual_persistence.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(np.arange(1,len(s)+1),s)
    ax[0].axvline(elbow,ls=':',color='gray')
    ax[0].set(xlabel='Rank',ylabel='Singular value',title=f'SVD spectrum (elbow ≈ {elbow})')
    ax[1].plot(np.arange(1,len(s)+1),100*cumulative)
    ax[1].axhline(args.variance_target*100,ls='--',color='gray')
    ax[1].axvline(threshold_rank,ls=':',color='gray')
    ax[1].set(xlabel='Rank',ylabel='Variance (%)',title=f'{args.variance_target:.0%} at rank {threshold_rank}')
    fig.tight_layout();fig.savefig(out/'svd_rank_selection.png',dpi=175);plt.close(fig)
    # First held-out day only to keep the visualization genuinely 78 x N.
    first_day=test_time[0][:10]
    ix=np.array([x[:10]==first_day for x in test_time])
    def panels(arrays,titles,filename):
        fig,axes=plt.subplots(1,len(arrays),figsize=(5*len(arrays),6))
        if len(arrays)==1:axes=[axes]
        for a,arr,label in zip(axes,arrays,titles):
            image=a.imshow(arr[ix].T,aspect='auto',cmap='RdBu_r',vmin=-3,vmax=3)
            a.set(title=label,xlabel='Five-minute bucket',ylabel='Stock')
        fig.colorbar(image,ax=axes.tolist(),shrink=.7)
        fig.suptitle(f'{args.metric} — held-out day {first_day}; provisional')
        fig.savefig(out/filename,dpi=160,bbox_inches='tight');plt.close(fig)
    panels([Ztest,low,residual],['Observed','SVD low rank','Residual'],'svd_components.png')
    panels([Ztest,low_rp,sparse,noise],['Observed','Robust low rank','Sparse','Error'],'rpca_components.png')
    summary={'metric':args.metric,'train_days':distinct[:train_days],
      'test_days':distinct[train_days:],'train_rows':int(train.sum()),'test_rows':int(test.sum()),
      'rank_used':k,'rank_80pct':threshold_rank,'elbow_heuristic':elbow,
      'explained_training_variance':float(cumulative[k-1]),
      'rpca_train_relative_error':float(err),'rpca_sparse_fraction':float(np.mean(S!=0)),
      'rpca_iteration_count':steps,'rpca_heldout_sparse_threshold':float(threshold),
      'warning':'Not validated against actual quotes; residual persistence is not price-return prediction.'}
    (out/'factor_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Created PNG and CSV outputs under:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('build')
    a.add_argument('--universe',default='universe_candidates.csv')
    a.add_argument('--data-root',default='/lobster/2025')
    a.add_argument('--dates',nargs='*')
    a.add_argument('--max-days',type=int)
    a.add_argument('--limit',type=int)
    a.add_argument('--half-life-days',type=float,default=5.0)
    a.add_argument('--output',default='team_results')
    b=sub.add_parser('factors')
    b.add_argument('--input',default='team_results')
    b.add_argument('--output',default='team_results/factors')
    b.add_argument('--metric',choices=('ofi_raw','ofi_per_volume','ofi_per_depth'),default='ofi_per_depth')
    b.add_argument('--train-days',type=int)
    b.add_argument('--half-life-days',type=float)
    b.add_argument('--variance-target',type=float,default=.8)
    b.add_argument('--rank',type=int)
    b.add_argument('--rpca-steps',type=int,default=150)
    args=p.parse_args()
    if args.cmd=='build':build(args)
    else:factors(args)


if __name__=='__main__':main()