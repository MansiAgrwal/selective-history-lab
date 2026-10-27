"""Experiment 3b: information ceiling. How narrow could the intervals be if measurement were perfect?

Uses the same simulated datasets as Experiment 3 (same seeds). The oracle observes every learner's latent state
before and after every practice trial and knows each learner's type, so the four gains are estimated by logistic
regression on the learning events themselves (unmastered skill practised -> mastered or not), with the known type
offset. Its interval width is a lower bound for any measurement design with these learners and this practice.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from history_model import Config, STATES, target
from history_model_ext import simulate_ext

REPS, N, H = 40, 120, 0.8
GAIN_INDEX = {(0, 0): 0, (0, 2): 1, (1, 1): 2, (1, 2): 3}   # (skill, activity) -> gain parameter: a|A, a|AB, b|B, b|AB


def oracle_fit(obs, truth, h):
    rows = []
    practice = obs.transition == 1
    for i, t in zip(*np.nonzero(practice)):
        s0, s1 = STATES[truth['before'][i, t]], STATES[truth['before'][i, t + 1]]
        e = obs.action[i, t]
        for k in (0, 1):
            if s0[k] == 0 and (k, e) in GAIN_INDEX:
                rows.append((GAIN_INDEX[(k, e)], truth['type'][i] * h, s1[k]))
    idx, off, y = map(np.asarray, zip(*rows))
    theta = logit(np.array([.06] * 4))
    for _ in range(50):                                   # Newton-Raphson, one intercept per gain with known offset
        p = expit(theta[idx] + off)
        grad = np.bincount(idx, y - p, 4); info = np.bincount(idx, p * (1 - p), 4)
        step = grad / info; theta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.diag(1 / info)
    est = target(expit(theta), h)
    J = np.zeros((2, 4)); eps = 1e-6
    for j in range(4):
        d = np.zeros(4); d[j] = eps
        J[:, j] = (target(expit(theta + d), h) - target(expit(theta - d), h)) / (2 * eps)
    se = np.sqrt(np.diag(J @ cov @ J.T))
    return est, se, np.bincount(idx, minlength=4)


if __name__ == '__main__':
    out = []
    for rep in range(REPS):
        obs, truth = simulate_ext(Config(n=N), 'recency', 9_000_000 + rep, 'none', 3)
        est, se, events = oracle_fit(obs, truth, H)
        for j, name in enumerate(('a: AB minus A', 'b: AB minus B')):
            out.append(dict(rep=rep, contrast=name, truth=truth['target'][j], estimate=est[j], width=3.92 * se[j],
                            covered=abs(est[j] - truth['target'][j]) <= 1.96 * se[j],
                            events_A_or_B=events[0 if j == 0 else 2], events_AB=events[1 if j == 0 else 3]))
    df = pd.DataFrame(out); df.to_csv('results/e3b_oracle.csv', index=False)
    print(df.groupby('contrast')[['width', 'covered', 'events_A_or_B', 'events_AB']].mean().round(4).to_string())
