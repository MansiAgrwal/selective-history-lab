"""Experiment 3: how reliable must a second channel be to beat simply asking more questions?

Every condition uses the same simulated learners, practice and latent trajectories (common random numbers):
N = 120, recency-retained history (capacity 18), heterogeneity 0.8, the pilot's gains. All datasets are fitted with
the specified-mixture model (heterogeneity known), as in the pilot, so the comparison isolates measurement.
Conditions:
  behaviour only with 9 (pilot), 18 or 27 probe items per probe block;
  behaviour + total-mastery sensor (one reading per probe block, as in the pilot), SD 0.2, 0.4 or 0.8;
  behaviour + skill-specific sensor (one reading of each skill per probe block), SD 0.2, 0.4 or 0.8.
The pilot's sensor is the total-mastery sensor at SD 0.8.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from history_model import Config
from history_model_ext import simulate_ext, fit_ext

REPS, N, H = 40, 120, 0.8
CONDITIONS = [('behaviour, 9 probe items', 'none', 3, .8), ('behaviour, 18 probe items', 'none', 6, .8),
              ('behaviour, 27 probe items', 'none', 9, .8)]
CONDITIONS += [(f'{kind} sensor, SD {sd}', kind, 3, sd) for kind in ('total', 'skill') for sd in (0.2, 0.4, 0.8)]
OUT = Path(__file__).resolve().parent / 'results' / 'e3_fits.jsonl'


def one(job):
    rep, (name, sensor, probe_reps, sd) = job
    seed = 9_000_000 + rep
    obs, truth = simulate_ext(Config(n=N, aux_sd=sd), 'recency', seed, sensor, probe_reps)
    r = fit_ext(obs, use_sensor=sensor != 'none', fixed_h=H, seed=seed, starts=2)
    return dict(rep=rep, condition=name, sensor=sensor, probe_items=3 * probe_reps, sensor_sd=sd if sensor != 'none' else None,
                truth=truth['target'].tolist(), **{k: r[k] for k in ('estimate', 'converged', 'boundary', 'interval_valid', 'ci_model', 'ci_sandwich')})


if __name__ == '__main__':
    done = set()
    if OUT.exists():
        done = {(r['rep'], r['condition']) for r in map(json.loads, OUT.read_text().splitlines())}
    jobs = [(rep, c) for rep in range(REPS) for c in CONDITIONS if (rep, c[0]) not in done]
    start = time.time()
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool, OUT.open('a') as f:
        futures = [pool.submit(one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result()) + '\n'); f.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f'{k}/{len(jobs)} fits, {time.time() - start:.0f}s', flush=True)
