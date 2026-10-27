"""Small Monte Carlo implementation pilot. Not a powered coverage study."""
import argparse, json, time, sys, platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.stats import beta
from history_model import Config, simulate, fit, score_heldout, loglik, start_theta

def run_one(rep,rule,scenario,n,starts):
    c=Config(n=n,aux_drift=0 if scenario=='invariant' else .75,
             aux_subject_sd=0 if scenario=='invariant' else .6)
    seed=20261025+1000*rep+(0 if rule=='recency' else 100)
    obs,truth=simulate(c,rule,seed)
    rows=[]
    for h in (0.,.8):
        for aux in (False,True):
            result=fit(obs,auxiliary=aux,fixed_h=h,seed=seed,starts=starts)
            rows.append(dict(rep=rep,rule=rule,scenario=scenario,n=n,
                model=('specified_mixture' if h else 'homogeneous')+('_joint' if aux else '_behavior'),
                truth=truth['target'].tolist(),**result))
    return rows

def summarize(rows):
    flat=[]
    for r in rows:
        for j,name in enumerate(('a_AB_minus_A','b_AB_minus_B')):
            q={k:r[k] for k in ('rep','rule','scenario','n','model','converged','boundary','interval_valid')}
            q.update(contrast=name,truth=r['truth'][j],estimate=r['estimate'][j],error=r['estimate'][j]-r['truth'][j])
            for kind in ('model','sandwich'):
                ci=r['ci_'+kind] if r['interval_valid'] else None
                q['coverage_'+kind]=float(ci[j][0]<=r['truth'][j]<=ci[j][1]) if ci else np.nan
                q['width_'+kind]=ci[j][1]-ci[j][0] if ci else np.nan
            flat.append(q)
    df=pd.DataFrame(flat);out=[]
    for keys,g in df.groupby(['scenario','rule','model','contrast']):
        n=len(g);nv=int(g.interval_valid.sum())
        d=dict(zip(['scenario','rule','model','contrast'],keys))
        d.update(repetitions=n,bias=g.error.mean(),bias_mcse=g.error.std(ddof=1)/np.sqrt(n),
            rmse=np.sqrt(np.mean(g.error**2)),converged_fraction=g.converged.mean(),
            boundary_fraction=g.boundary.mean(),valid_interval_fraction=nv/n,valid_intervals=nv)
        for kind in ('model','sandwich'):
            cov=g['coverage_'+kind].mean()
            d['conditional_coverage_'+kind]=cov
            d['coverage_mcse_'+kind]=np.sqrt(cov*(1-cov)/nv) if nv else np.nan
            # Exact binomial bounds remain informative when plug-in MCSE is zero.
            k=int(g['coverage_'+kind].fillna(0).sum())
            d['coverage_mc95_low_'+kind]=float(beta.ppf(.025,k,nv-k+1)) if k else 0.
            d['coverage_mc95_high_'+kind]=float(beta.ppf(.975,k+1,nv-k)) if k<nv else 1.
            d['valid_and_cover_fraction_'+kind]=g['coverage_'+kind].fillna(0).mean()
            d['mean_width_'+kind]=g['width_'+kind].mean()
        out.append(d)
    return df,pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=12)
    ap.add_argument('--n',type=int,default=120)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--starts',type=int,default=2)
    ap.add_argument('--out',default='outputs/pilot')
    ap.add_argument('--resume',action='store_true',help='Resume complete four-model dataset groups from fits.json')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.time();allrows=[]
    if args.resume and (out/'fits.json').exists():
        allrows=json.loads((out/'fits.json').read_text())
        if any(r['n']!=args.n or r['starts']!=args.starts for r in allrows):
            raise ValueError('Resume requires the same N and optimizer starts')
        counts={}
        for r in allrows:
            k=(r['rep'],r['rule'],r['scenario']);counts[k]=counts.get(k,0)+1
        if any(v!=4 for v in counts.values()):raise ValueError('Incomplete dataset group in checkpoint')
    completed={(r['rep'],r['rule'],r['scenario']) for r in allrows}
    # Compile once before concurrent optimization.
    warm,_=simulate(Config(n=1,blocks=1));loglik(start_theta(),warm)
    jobs=[(r,p,s,args.n,args.starts) for r in range(args.reps)
          for p in ('recency','balanced') for s in ('invariant','drift_and_subject_noise')
          if (r,p,s) not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(run_one,*job) for job in jobs]
        for k,f in enumerate(as_completed(pending),1):
            allrows.extend(f.result())
            (out/'fits.json').write_text(json.dumps(allrows,indent=2))
            print(f'{k}/{len(jobs)} datasets; {len(allrows)} fits; {time.time()-started:.0f}s',flush=True)
    allrows.sort(key=lambda r:(r['scenario'],r['rule'],r['rep'],r['model']))
    (out/'fits.json').write_text(json.dumps(allrows,indent=2))
    flat,summary=summarize(allrows)
    flat.to_csv(out/'replicate_results.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    meta={'status':'implementation pilot, not a powered coverage or sample-size study',
          'arguments':vars(args),'seconds_this_invocation':time.time()-started,'python':sys.version,
          'platform':platform.platform(),'base_config':asdict(Config(n=args.n)),
          'known_to_fitter':['retention loss','equal mixture weights and specified heterogeneity support'],
          'not_implemented':['human confidence','feature attention and generalization','fMRI haemodynamics','estimated heterogeneity','bootstrap intervals']}
    (out/'run_manifest.json').write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
