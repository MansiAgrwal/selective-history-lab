"""Illustration of transport to an independent, uniformly assigned test cohort.

Fits are taken from replicate zero. Test data never change fitted parameters.
One test cohort per scenario is shared across models and training policies.
This is an illustrative test, not a Monte Carlo calibration comparison.
"""
import json
from pathlib import Path
import pandas as pd
from history_model import Config,simulate,score_heldout

def main():
    root=Path(__file__).resolve().parent
    fits=json.loads((root/'outputs/pilot/fits.json').read_text());rows=[]
    for scenario in ('invariant','drift_and_subject_noise'):
        c=Config(n=240,aux_drift=0 if scenario=='invariant' else .75,
                 aux_subject_sd=0 if scenario=='invariant' else .6)
        obs,_=simulate(c,'random',seed=98261025)
        for r in fits:
            if r['rep'] or r['scenario']!=scenario:continue
            s=score_heldout(r['theta'],obs,27,r['auxiliary'],r['fixed_h'])
            rows.append({'scenario':scenario,'training_policy':r['rule'],'model':r['model'],
                         'training_converged':r['converged'],'test_policy':'uniform_random',**s})
    out=root/'outputs/prediction';out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out/'independent_cohort.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
