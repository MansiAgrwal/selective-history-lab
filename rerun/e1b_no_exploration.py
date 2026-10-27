"""Experiment 1b: same comparison with the exploration floor removed (epsilon = 0), capacities 3 and 18."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import numpy as np, pandas as pd
from history_model import Config
from history_model_ext import simulate_ext
from e1_manipulation import outcomes, N, SEED, OUT
rows = []
for cap in (3, 18):
    r = outcomes(*simulate_ext(Config(n=N, capacity=cap, epsilon=0.0), 'recency', SEED, 'none'))
    b = outcomes(*simulate_ext(Config(n=N, capacity=cap, epsilon=0.0), 'balanced', SEED, 'none'))
    d = b - r
    rows.append(dict(capacity=cap, epsilon=0.0, **{k: d[k].mean() for k in ('n_A', 'n_B', 'n_AB')},
                     mastered_both=d.mastered_both.mean(), mastered_both_se=d.mastered_both.std(ddof=1) / np.sqrt(N),
                     delayed_acc=d.delayed_acc.mean(), delayed_acc_se=d.delayed_acc.std(ddof=1) / np.sqrt(N)))
out = pd.DataFrame(rows); out.to_csv(OUT / 'e1b_no_exploration.csv', index=False); print(out.round(4).to_string(index=False))
