"""Experiment 1b: the retention-rule comparison with exploration removed in two steps, capacities 3, 6, 9 and 18.

(1) epsilon = 0: removes the uniform mixture from the adaptive choices, but each 12-trial block still schedules one
A, one B and one AB trial (18 of the 90 practice trials), and the 18 initial practice trials are fixed.
(2) epsilon = 0 and no scheduled exploration: all 72 block trials are adaptive. Only the 18 fixed initial practice
trials remain non-adaptive. Same 2,000 learners as Experiment 1; differences are balanced minus recency, paired by learner.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import numpy as np, pandas as pd
from history_model import Config
from history_model_ext import simulate_ext
from e1_manipulation import outcomes, N, SEED, OUT
rows = []
for scheduled in (True, False):
    for cap in (3, 6, 9, 18):
        c = Config(n=N, capacity=cap, epsilon=0.0)
        r = outcomes(*simulate_ext(c, 'recency', SEED, 'none', scheduled_exploration=scheduled))
        b = outcomes(*simulate_ext(c, 'balanced', SEED, 'none', scheduled_exploration=scheduled))
        d = b - r
        rows.append(dict(capacity=cap, epsilon=0.0, scheduled_exploration=scheduled,
                         **{k: d[k].mean() for k in ('n_A', 'n_B', 'n_AB')},
                         mastered_both=d.mastered_both.mean(), mastered_both_se=d.mastered_both.std(ddof=1) / np.sqrt(N),
                         delayed_acc=d.delayed_acc.mean(), delayed_acc_se=d.delayed_acc.std(ddof=1) / np.sqrt(N)))
out = pd.DataFrame(rows); out.to_csv(OUT / 'e1b_no_exploration.csv', index=False); print(out.round(4).to_string(index=False))
