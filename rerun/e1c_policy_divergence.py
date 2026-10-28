"""Experiment 1c: how far apart are the two retention rules' assignment probabilities?

(a) A random search over constructed practice histories (18-90 trials, varied activity mix and success rates, half
ending in a run of one activity) for the largest difference in P(assign A), capacity 18, epsilon 0.10.
(b) The decisions simulated learners actually reach: for 500 recency learners (Experiment 1 seed), at every adaptive
decision, both rules are applied to the same logged history and the total-variation distance between their
assignment probabilities is recorded.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import numpy as np, pandas as pd
from history_model import policy, Config
from history_model_ext import simulate_ext
from e1_manipulation import SEED, OUT

rng = np.random.default_rng(0)
best = (0., None)
for _ in range(200_000):
    L = int(rng.integers(18, 91))
    mix = rng.dirichlet(np.ones(3) * rng.uniform(.2, 3)); success = rng.uniform(0, 1, 3)
    acts = rng.choice(3, L, p=mix)
    if rng.random() < .5:
        k = int(rng.integers(6, 19)); acts[-k:] = int(rng.integers(3))
    hist = [(int(a), int(rng.random() < success[a])) for a in acts]
    d = policy(hist, 'balanced', 18, .1)[0] - policy(hist, 'recency', 18, .1)[0]
    if abs(d) > abs(best[0]):
        best = (d, hist)
rows = [dict(analysis='constructed histories (search)', capacity=18, decisions=200_000,
             p_A_recency=policy(best[1], 'recency', 18, .1)[0], p_A_balanced=policy(best[1], 'balanced', 18, .1)[0],
             max_abs_difference_p_A=abs(best[0]))]
for cap in (3, 18):
    c = Config(n=500, capacity=cap)
    obs, _ = simulate_ext(c, 'recency', SEED, 'none')
    tv = []
    for i in range(c.n):
        prac = obs.transition[i] == 1
        acts, ys, phs = obs.action[i][prac], obs.response[i][prac], obs.phase[i][prac]
        for t in np.where(phs == 3)[0]:
            h = list(zip(acts[:t].tolist(), ys[:t].tolist()))
            tv.append(.5 * np.abs(policy(h, 'balanced', cap, .1) - policy(h, 'recency', cap, .1)).sum())
    tv = np.asarray(tv)
    rows.append(dict(analysis='decisions reached by simulated recency learners', capacity=cap, decisions=len(tv),
                     tv_median=np.median(tv), tv_p90=np.quantile(tv, .9), tv_p99=np.quantile(tv, .99), tv_max=tv.max()))
out = pd.DataFrame(rows); out.to_csv(OUT / 'e1c_policy_divergence.csv', index=False); print(out.round(3).to_string(index=False))
