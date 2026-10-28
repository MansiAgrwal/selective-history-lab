"""Experiment 3d: does placement matter when the number of readings is held fixed?

Experiment 3c read the channel on all 90 practice trials, against 7 (total) or 14 (skill) readings at the probe
blocks, so it changed timing and quantity together. Here the readings are as many as in Experiment 3 but placed on
practice trials: per practice segment (the 18 initial trials and each 12-trial block), one reading of the total
sensor, or two of the skill sensor, at trials drawn uniformly within the segment. Each reading is taken before that
trial's learning. Same learners, practice and latent trajectories as Experiments 3 and 3c (replications 0-19),
fitted with the specified-mixture model. Compared, paired by replication, with the probe-time rows of Experiment 3.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from history_model import Config
from history_model_ext import simulate_ext, fit_ext

REPS, N, H = 20, 120, 0.8
CONDITIONS = [(f'{kind} sensor on {n} practice trials, SD {sd}', kind, sd) for kind, n in (('total', 7), ('skill', 14)) for sd in (0.4, 0.8)]
OUT = Path(__file__).resolve().parent / 'results' / 'e3d_fits.jsonl'


def one(job):
    rep, (name, sensor, sd) = job
    seed = 9_000_000 + rep
    obs, truth = simulate_ext(Config(n=N, aux_sd=sd), 'recency', seed, sensor, 3, sensor_at='practice_sparse')
    r = fit_ext(obs, use_sensor=True, fixed_h=H, seed=seed, starts=2)
    return dict(rep=rep, condition=name, sensor=sensor, probe_items=9, sensor_sd=sd, readings=int((obs.zkind > 0).sum() // N),
                truth=truth['target'].tolist(), **{k: r[k] for k in ('estimate', 'converged', 'boundary', 'interval_valid', 'ci_model', 'ci_sandwich')})


if __name__ == '__main__':
    done = set()
    if OUT.exists():
        done = {(r['rep'], r['condition']) for r in map(json.loads, OUT.read_text().splitlines())}
    jobs = [(rep, c) for rep in range(REPS) for c in CONDITIONS if (rep, c[0]) not in done]
    start = time.time()
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool, OUT.open('a') as f:
        for k, fut in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            f.write(json.dumps(fut.result()) + '\n'); f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f'{k}/{len(jobs)} fits, {time.time() - start:.0f}s', flush=True)
