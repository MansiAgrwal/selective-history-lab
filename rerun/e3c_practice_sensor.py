"""Experiment 3c: does a second channel help if it is measured on every practice trial instead of at probe blocks?

Same simulated learners, practice and latent trajectories as Experiment 3 (same seeds, replications 0-19). One sensor
reading on each of the 90 practice trials, taken before that trial's learning transition. Fitted with the
specified-mixture model, as in Experiment 3. This is not an equal-burden comparison: 90 readings per person
instead of 7.
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
CONDITIONS = [(f'{kind} sensor on every practice trial, SD {sd}', kind, sd) for kind in ('total', 'skill') for sd in (0.4, 0.8)]
OUT = Path(__file__).resolve().parent / 'results' / 'e3c_fits.jsonl'


def one(job):
    rep, (name, sensor, sd) = job
    seed = 9_000_000 + rep
    obs, truth = simulate_ext(Config(n=N, aux_sd=sd), 'recency', seed, sensor, 3, sensor_at='practice')
    r = fit_ext(obs, use_sensor=True, fixed_h=H, seed=seed, starts=2)
    return dict(rep=rep, condition=name, sensor=sensor, probe_items=9, sensor_sd=sd, truth=truth['target'].tolist(),
                **{k: r[k] for k in ('estimate', 'converged', 'boundary', 'interval_valid', 'ci_model', 'ci_sandwich')})


if __name__ == '__main__':
    done = set()
    if OUT.exists():
        done = {(r['rep'], r['condition']) for r in map(json.loads, OUT.read_text().splitlines())}
    jobs = [(rep, c) for rep in range(REPS) for c in CONDITIONS if (rep, c[0]) not in done]
    start = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool, OUT.open('a') as f:
        for k, fut in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            f.write(json.dumps(fut.result()) + '\n'); f.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f'{k}/{len(jobs)} fits, {time.time() - start:.0f}s', flush=True)
