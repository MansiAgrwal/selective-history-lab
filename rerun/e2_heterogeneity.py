"""Experiment 2: when does ignoring persistent learner heterogeneity distort the target contrasts?

Grid: heterogeneity h in {0.8, 1.6, 2.4} x gain level {1x, 3x the pilot's base gains}. The personalization
algorithm uses recency-retained history (capacity 18). Each simulated dataset (N = 120) is fitted twice with
the original model: homogeneous (h = 0, misspecified whenever h > 0) and specified mixture (h known, an oracle
benchmark). Common random numbers: replicate r uses the same seed in every cell.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from history_model import Config, simulate, fit

H, SCALES, REPS, N = (0.8, 1.6, 2.4), (1, 3), 60, 120
BASE = np.array((.08, .045, .055, .04))
OUT = Path(__file__).resolve().parent / 'results' / 'e2_fits.jsonl'

def one(job):
    rep, h, scale = job
    c = Config(n=N, gains=tuple(float(g) for g in BASE * scale), heterogeneity=h)
    seed = 7_000_000 + rep
    obs, truth = simulate(c, 'recency', seed)
    rows = []
    for fixed_h, name in ((0.0, 'homogeneous'), (h, 'specified_mixture')):
        r = fit(obs, auxiliary=False, fixed_h=fixed_h, seed=seed, starts=2)
        rows.append(dict(rep=rep, h=h, gain_scale=scale, model=name, truth=truth['target'].tolist(),
                         **{k: r[k] for k in ('estimate', 'converged', 'boundary', 'interval_valid', 'ci_model', 'ci_sandwich', 'nll', 'same_best_starts')}))
    return rows

if __name__ == '__main__':
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line); done.add((r['rep'], r['h'], r['gain_scale']))
    jobs = [(rep, h, s) for rep in range(REPS) for h in H for s in SCALES if (rep, h, s) not in done]
    start = time.time()
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool, OUT.open('a') as f:
        futures = [pool.submit(one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                f.write(json.dumps(row) + '\n')
            f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f'{k}/{len(jobs)} datasets, {time.time() - start:.0f}s', flush=True)
