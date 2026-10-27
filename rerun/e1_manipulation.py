"""Experiment 1: does the history manipulation change what learners practise, and does that change learning?

The same 2,000 simulated learners (same persistent type and initial state) go through every condition:
recency vs balanced retained history at capacities 3, 6, 9 and 18; uniform random assignment; and a yoked arm
in which each learner receives the practice sequence the recency algorithm generated for a different person.
Differences are paired by learner, so their standard errors are small. No model fitting.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from history_model import Config
from history_model_ext import simulate_ext

N, SEED, PARTNER_SEED = 2000, 101, 202
OUT = Path(__file__).resolve().parent / 'results'


def outcomes(obs, truth):
    pa = truth['practice_actions']
    counts = np.stack([(pa == e).sum(1) for e in range(3)], 1)
    end_practice = truth['before'][np.arange(len(pa)), np.argmax(obs.phase == 4, axis=1)]   # state at the first immediate-test item
    imm = np.where(obs.phase == 4, obs.response, 0).sum(1) / (obs.phase == 4).sum(1)
    dly = np.where(obs.phase == 6, obs.response, 0).sum(1) / (obs.phase == 6).sum(1)
    return pd.DataFrame(dict(n_A=counts[:, 0], n_B=counts[:, 1], n_AB=counts[:, 2],
                             mastered_a=(end_practice % 2 == 1).astype(float), mastered_b=(end_practice >= 2).astype(float),
                             mastered_both=(end_practice == 3).astype(float), immediate_acc=imm, delayed_acc=dly))


def run():
    conditions = {}
    for cap in (3, 6, 9, 18):
        for rule in ('recency', 'balanced'):
            conditions[f'{rule}_cap{cap}'] = outcomes(*simulate_ext(Config(n=N, capacity=cap), rule, SEED, 'none'))
    conditions['random'] = outcomes(*simulate_ext(Config(n=N), 'random', SEED, 'none'))
    partner = simulate_ext(Config(n=N), 'recency', PARTNER_SEED, 'none')[1]['practice_actions']
    conditions['yoked_to_recency_cap18'] = outcomes(*simulate_ext(Config(n=N), 'recency', SEED, 'none', yoked=partner))

    rows = []
    for name, d in conditions.items():
        rows.append(dict(condition=name, **{k: d[k].mean() for k in d.columns}))
    means = pd.DataFrame(rows)

    def paired(a, b):
        out = {}
        for k in conditions[a].columns:
            diff = conditions[a][k] - conditions[b][k]
            out[k] = diff.mean(); out[k + '_se'] = diff.std(ddof=1) / np.sqrt(len(diff))
        comp_a = conditions[a][['n_A', 'n_B', 'n_AB']].to_numpy(); comp_b = conditions[b][['n_A', 'n_B', 'n_AB']].to_numpy()
        tv = .5 * np.abs(comp_a / comp_a.sum(1, keepdims=True) - comp_b / comp_b.sum(1, keepdims=True)).sum(1)
        out['per_learner_composition_tv'] = tv.mean()
        return out

    contrasts = [(f'balanced_cap{c}', f'recency_cap{c}') for c in (3, 6, 9, 18)]
    contrasts += [('random', 'recency_cap18'), ('yoked_to_recency_cap18', 'recency_cap18')]
    diffs = pd.DataFrame([dict(contrast=f'{a} minus {b}', **paired(a, b)) for a, b in contrasts])
    means.to_csv(OUT / 'e1_condition_means.csv', index=False)
    diffs.to_csv(OUT / 'e1_paired_differences.csv', index=False)
    return means, diffs


if __name__ == '__main__':
    means, diffs = run()
    pd.set_option('display.width', 220)
    print(means.round(3).to_string(index=False))
    show = ['contrast', 'n_A', 'n_B', 'n_AB', 'per_learner_composition_tv', 'mastered_both', 'mastered_both_se', 'delayed_acc', 'delayed_acc_se']
    print(diffs[show].round(4).to_string(index=False))
