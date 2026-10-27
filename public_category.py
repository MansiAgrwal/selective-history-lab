"""Audit and descriptive baseline for Mack, Love & Preston (2016), OSF 5byhb.

No imaging is loaded; this is not a causal adaptive-policy or neural replication.
The redistributed source files are CC BY 4.0; see data/public_mack/ATTRIBUTION.md.
"""
import argparse, hashlib, io, json, re, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCES={'description.txt':'https://osf.io/download/57sxd/',
         'featureinfo.txt':'https://osf.io/download/jt5we/',
         'behaviour.zip':'https://osf.io/download/ymdfp/',
         'trialtiming.zip':'https://osf.io/download/25sj5/'}

def acquire(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name,url in SOURCES.items():
        p=folder/name
        if not p.exists():
            with urllib.request.urlopen(url,timeout=60) as r:data=r.read()
            if name.endswith('.zip') and not data.startswith(b'PK'):
                raise ValueError('Expected a ZIP archive, received different content')
            p.write_bytes(data)
    return folder

def load_data(folder):
    behavior=[];timing=[]
    columns=['trial','task','run','dim1','dim2','dim3','answer','response','rt_ms','accuracy']
    with zipfile.ZipFile(folder/'behaviour.zip') as z:
        names=[n for n in z.namelist() if re.search(r'/\d+_study\d_run\d\.txt$',n) and '__MACOSX' not in n]
        for name in sorted(names):
            sub=int(re.search(r'/(\d+)_study',name)[1])
            f=pd.read_csv(io.BytesIO(z.read(name)),sep='\t',header=None,names=columns)
            if f.shape!=(32,10):raise ValueError(f'Unexpected shape: {name}: {f.shape}')
            f.insert(0,'subject',sub);behavior.append(f)
    with zipfile.ZipFile(folder/'trialtiming.zip') as z:
        names=[n for n in z.namelist() if re.search(r'/\d+_study\d_run\d\.txt$',n) and '__MACOSX' not in n]
        for name in sorted(names):
            timing.append(pd.read_csv(io.BytesIO(z.read(name)),sep='\t',header=None,
                names=['subject','trial','task','run','stimulus_onset_s','feedback_onset_s']))
    b=pd.concat(behavior,ignore_index=True);t=pd.concat(timing,ignore_index=True)
    keys=['subject','task','run','trial']
    d=b.merge(t,on=keys,how='outer',validate='one_to_one',indicator=True)
    if not (d['_merge']=='both').all():raise ValueError('Timing/behavior records do not match')
    d=d.drop(columns='_merge').sort_values(keys).reset_index(drop=True)
    # The raw task/study code identifies the rule, not chronological period.
    # Stimulus-answer mappings independently verify 1->VI, 2->I, 3->II.
    d['rule']=d.task.map({1:6,2:1,3:2})
    d['period']=np.where((d.subject%2==1)&(d.task>1),5-d.task,d.task)
    d['trial_in_rule']=(d.run-1)*32+d.trial
    d['global_run']=(d.period-1)*4+d.run
    expected=np.where(d.rule==1,d.dim1+1,np.where(d.rule==2,(d.dim2+d.dim3)%2+1,
                      (d.dim1+d.dim2+d.dim3)%2+1))
    if not (d.answer==expected).all():raise ValueError('Category labels disagree with documentation')
    if not (d.accuracy==(d.response==d.answer).astype(int)).all():raise ValueError('Accuracy does not match responses')
    if d.duplicated(keys).any():raise ValueError('Duplicate trials')
    counts=d.groupby(['subject','task','run','dim1','dim2','dim3']).size()
    if not (counts==4).all():raise ValueError('Exemplar counts changed: expected four each')
    if not (d.feedback_onset_s>d.stimulus_onset_s).all():raise ValueError('Invalid timing order')
    return d.sort_values(['subject','period','run','trial']).reset_index(drop=True)

def run(folder,out):
    folder=acquire(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=load_data(folder);d.to_csv(out/'events.csv',index=False)
    rng=np.random.default_rng(20261025);rows=[]
    for sample,dd in [('all_behavior',d),('exclude_subject_09',d[d.subject!=9])]:
        for (rule,run),g in dd.groupby(['rule','run']):
            a=g.groupby('subject').accuracy.mean().to_numpy()
            boot=np.mean(rng.choice(a,size=(2000,len(a)),replace=True),axis=1)
            rows.append({'sample':sample,'rule':int(rule),'run':int(run),'participants':len(a),
                'accuracy':float(a.mean()),'lower':float(np.quantile(boot,.025)),
                'upper':float(np.quantile(boot,.975))})
    summary=pd.DataFrame(rows);summary.to_csv(out/'learning_curves.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.4,4.3),layout='constrained')
    for rule,color in [(1,'#176b9b'),(2,'#bc7423'),(6,'#73549b')]:
        s=summary[(summary['sample']=='all_behavior')&(summary.rule==rule)]
        ax.plot(s.run,s.accuracy,'o-',color=color,label=f'Type {rule}')
        ax.fill_between(s.run,s.lower,s.upper,color=color,alpha=.12)
    ax.set(xlabel='Run within category rule (32 trials per run)',ylabel='Mean accuracy',
           ylim=(.35,1.02),xticks=[1,2,3,4],title='Public category-learning baseline: 23 participants')
    ax.axhline(.5,color='.6',linestyle=':',linewidth=1);ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    checks={'participants':int(d.subject.nunique()),'trials':len(d),'runs':int(d.groupby(['subject','task','run']).ngroups),
        'trials_per_run':32,'repetitions_per_exemplar':4,'omissions':int(d.response.isna().sum()),
        'missing_rt':int(d.rt_ms.isna().sum()),'label_mismatches':0,'timing_join':'one-to-one, all rows matched',
        'accuracy_includes_omissions_as_incorrect':True,'subject09_in_behavior':True,
        'sensitivity_excludes_subject09':True,'type6_always_first':True,
        'task_code_is_chronology':False,'task_code_to_rule':{'1':6,'2':1,'3':2},
        'period_mapping':'even subjects VI/I/II; odd subjects VI/II/I, per source README',
        'neural_files_processed':0,'confidence_measure_available':False,
        'inference':'descriptive, not an adaptive-policy causal effect',
        'bootstrap':'2,000 participant resamples per rule/run; pointwise percentile intervals, not simultaneous',
        'documentation_discrepancy':'README says three repetitions; files contain four per eight stimuli in each 32-trial run'}
    (out/'audit.json').write_text(json.dumps(checks,indent=2))
    (out/'provenance.json').write_text(json.dumps({'retrieved':'2026-10-25','project':'https://osf.io/5byhb/',
        'authors':'Mack, Love & Preston (2016)','license':'CC BY 4.0',
        'files':[{'name':k,'url':v,'sha256':hashlib.sha256((folder/k).read_bytes()).hexdigest(),
                  'bytes':(folder/k).stat().st_size} for k,v in SOURCES.items()]},indent=2))
    print(json.dumps(checks,indent=2));return d,summary,checks

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/public_mack');ap.add_argument('--out',default='outputs/public_category')
    a=ap.parse_args();run(a.data,a.out)
