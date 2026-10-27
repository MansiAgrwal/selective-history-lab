"""Extensions to history_model.py for the rerun. The original module is imported unchanged.

1. Per-learner random-number streams for (a) learner type, initial state, practice responses, transitions and the
   personalization algorithm's choices, (b) probe and test responses and (c) sensor noise. Conditions that differ
   only in the assessment or sensor design therefore share identical latent trajectories and practice, and
   learner i has the same type and initial state in every condition (common random numbers).
2. Variable probe length: probe_reps repetitions of (A, B, AB) per probe block; the pilot used 3 (9 items).
3. Sensor channels: 'none'; 'total' (one reading of a + b per probe block, as in the pilot); or 'skill'
   (two readings per probe block, one of skill a and one of skill b, sharing intercept, loading and noise).
4. Yoked assignment: replay a partner's practice sequence regardless of the learner's own responses.
5. sensor_at='practice': one sensor reading on every practice trial, taken before that trial's learning
   transition (skill sensor: skill a on A trials, skill b on B trials, total mastery on AB trials), instead of
   readings at the probe blocks.

The fitted models keep the pilot's assumptions: equal type weights, specified heterogeneity support, known
overnight loss.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import math
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from history_model import Config, STATES, REQ, policy, transition_matrix, initial_dist, target, pack, start_theta, njit

SENSOR_KIND = {'none': 0, 'total': 1, 'skill_a': 2, 'skill_b': 3}


@dataclass
class ObservedExt:
    action: np.ndarray
    response: np.ndarray
    transition: np.ndarray    # 0 identity, 1 practice, 2 overnight
    auxiliary: np.ndarray     # sensor value or NaN
    zkind: np.ndarray         # 0 none, 1 total mastery, 2 skill a, 3 skill b
    phase: np.ndarray         # 0 initial practice, 1 probe, 2 scheduled, 3 adaptive, 4 immediate test, 5 overnight, 6 delayed test
    assigned_probability: np.ndarray


def simulate_ext(c=Config(), rule='recency', seed=1, sensor='total', probe_reps=3, yoked=None, sensor_at='probes'):
    """Simulate N learners. `yoked` is an (n, n_practice) array of practice actions to replay."""
    if rule not in ('recency', 'balanced', 'random'):
        raise ValueError(rule)
    if sensor not in ('none', 'total', 'skill'):
        raise ValueError(sensor)
    if sensor_at not in ('probes', 'practice'):
        raise ValueError(sensor_at)
    if sensor == 'skill' and probe_reps < 1:
        raise ValueError('skill sensor needs at least two probe items per block')
    n_probe = 3 * probe_reps
    n_practice = 18 + 12 * c.blocks
    tmax = 18 + n_probe + c.blocks * (12 + n_probe) + 24 + 1 + 24
    shape = (c.n, tmax)
    a = np.zeros(shape, np.int64); y = np.full(shape, -1, np.int64); tr = np.zeros(shape, np.int64)
    z = np.full(shape, np.nan); zk = np.zeros(shape, np.int64); ph = np.zeros(shape, np.int64); pr = np.ones(shape)
    before = np.zeros(shape, np.int64); types = np.zeros(c.n); practice_actions = np.zeros((c.n, n_practice), np.int64)
    for i in range(c.n):
        main = np.random.default_rng([seed, i, 0]); assess = np.random.default_rng([seed, i, 1]); noise = np.random.default_rng([seed, i, 2])
        u = main.choice([-1, 1]); types[i] = u
        mats = [transition_matrix(c.gains, c.losses, e, u * c.heterogeneity) for e in range(3)]
        s = int(main.choice(4, p=initial_dist(c.initial_mastery)))
        hist, t, k_practice = [], 0, 0

        def event(e, kind, phase, p=1.0, zkind=0):
            nonlocal s, t, k_practice
            a[i, t] = e; tr[i, t] = kind; ph[i, t] = phase; pr[i, t] = p; before[i, t] = s
            if kind == 2:
                keep = main.random(2) >= np.asarray(c.retention_loss)
                cur = STATES[s] * keep; s = int(cur[0] + 2 * cur[1])
            else:
                eta = int(np.all(STATES[s] >= REQ[e]))
                rng = main if kind == 1 else assess
                y[i, t] = int(rng.random() < c.guess + (1 - c.slip - c.guess) * eta)
                if zkind:
                    level = {1: STATES[s].sum(), 2: STATES[s][0], 3: STATES[s][1]}[zkind]
                    z[i, t] = c.aux_loading * level + noise.normal(0, c.aux_sd); zk[i, t] = zkind
                if kind == 1:
                    hist.append((e, int(y[i, t]))); practice_actions[i, k_practice] = e; k_practice += 1
                    s = int(main.choice(4, p=mats[e][s]))
            t += 1

        def practice(e_default, phase, p=1.0):
            e = int(yoked[i, k_practice]) if yoked is not None else e_default
            e = int(e)
            zkind = 0
            if sensor != 'none' and sensor_at == 'practice':
                zkind = 1 if sensor == 'total' else {0: 2, 1: 3, 2: 1}[e]
            event(e, 1, phase, 1.0 if yoked is not None else p, zkind=zkind)

        def probe_block():
            for k, e in enumerate(np.tile(np.arange(3), probe_reps)):
                kind = 0
                if sensor_at == 'practice':
                    event(int(e), 0, 1); continue
                if sensor == 'total' and k == 0: kind = 1
                if sensor == 'skill' and k == 0: kind = 2
                if sensor == 'skill' and k == 1: kind = 3
                event(int(e), 0, 1, zkind=kind)

        initial = np.repeat(np.arange(3), [6, 6, 6] if c.initial == 'balanced' else [12, 3, 3]); main.shuffle(initial)
        for e in initial:
            practice(int(e), 0)
        probe_block()
        for block in range(c.blocks):
            positions = main.choice(12, 3, replace=False); order = main.permutation(3); scheduled = dict(zip(positions, order))
            for j in range(12):
                if j in scheduled:
                    practice(int(scheduled[j]), 2)
                else:
                    pi = policy(hist, rule, c.capacity, c.epsilon); e = int(main.choice(3, p=pi))
                    practice(e, 3, float(pi[e]))
            probe_block()
        for e in np.tile(np.arange(3), 8):
            event(int(e), 0, 4)
        event(0, 2, 5)
        for e in np.tile(np.arange(3), 8):
            event(int(e), 0, 6)
        assert t == tmax and k_practice == n_practice
    obs = ObservedExt(a, y, tr, z, zk, ph, pr)
    return obs, {'type': types, 'before': before, 'practice_actions': practice_actions,
                 'target': target(c.gains, c.heterogeneity)}


@njit(cache=True, nogil=True)
def _forward_ext(a, y, tr, z, zk, mats, initial, response, retention, mu, sd, use_z):
    out = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        f = np.empty((2, 4))
        for u in range(2):
            for s in range(4):
                f[u, s] = .5 * initial[s]
        ll = 0.
        for t in range(a.shape[1]):
            e = a[i, t]
            logs = np.zeros(4)
            for s in range(4):
                if y[i, t] >= 0:
                    pp = response[e, s]
                    logs[s] = math.log(pp if y[i, t] == 1 else 1 - pp)
                if use_z and zk[i, t] > 0:
                    logs[s] += -.5 * ((z[i, t] - mu[zk[i, t], s]) / sd) ** 2 - math.log(sd) - .5 * math.log(2 * math.pi)
            mx = np.max(logs); total = 0.
            for u in range(2):
                for s in range(4):
                    f[u, s] *= math.exp(logs[s] - mx); total += f[u, s]
            if total <= 0:
                return np.full(a.shape[0], -1e100)
            ll += math.log(total) + mx
            f /= total
            if tr[i, t] > 0:
                nxt = np.zeros((2, 4))
                for u in range(2):
                    for s in range(4):
                        for sp in range(4):
                            pp = mats[u, e, s, sp] if tr[i, t] == 1 else retention[s, sp]
                            nxt[u, sp] += f[u, s] * pp
                f = nxt
        out[i] = ll
    return out


def loglik_ext(theta, obs, use_sensor, fixed_h=0., retention_loss=(.15, .15)):
    mats, p, response, retention, _, _ = pack(theta[:10], False, fixed_h, retention_loss)
    mu = np.zeros((4, 4)); sd = 1.
    if use_sensor:
        c0, load, sd = theta[10], np.exp(theta[11]), np.exp(theta[12])   # np.exp, as in history_model.pack
        mu[1] = c0 + load * STATES.sum(axis=1); mu[2] = c0 + load * STATES[:, 0]; mu[3] = c0 + load * STATES[:, 1]
    return _forward_ext(obs.action, obs.response, obs.transition, obs.auxiliary, obs.zkind,
                        mats, p, response, retention, mu, sd, use_sensor)


def fit_ext(obs, use_sensor=False, fixed_h=0., retention_loss=(.15, .15), seed=1, starts=2, maxiter=350):
    """Same optimizer, bounds, interval rules and delta method as history_model.fit."""
    rng = np.random.default_rng(seed)
    objective = lambda x: -float(loglik_ext(x, obs, use_sensor, fixed_h, retention_loss).sum())
    bounds = [(-7, 1)] * 4 + [(-9, -.5)] * 2 + [(-5, 3)] * 2 + [(-5, 3)] * 2
    if use_sensor:
        bounds += [(-4, 4), (-4, 2), (-3, 3)]
    x0 = start_theta(use_sensor); candidates = []
    for j in range(starts):
        x = x0 if j == 0 else x0 + rng.normal(0, .55, len(x0))
        candidates.append(minimize(objective, x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-5}))
    finite = [r for r in candidates if np.isfinite(r.fun)]
    best = min(finite, key=lambda r: r.fun); x = best.x
    est = target(expit(x[:4]), fixed_h)
    boundary = any(min(v - lo, hi - v) < .02 for v, (lo, hi) in zip(x, bounds))
    answer = dict(estimate=est.tolist(), theta=x.tolist(), nll=float(best.fun), converged=bool(best.success),
                  boundary=bool(boundary), interval_valid=False, ci_model=None, ci_sandwich=None)
    d = len(x); h = 3e-4 * np.maximum(1, np.abs(x)); H = np.zeros((d, d)); f0 = objective(x)
    for j in range(d):
        ej = np.zeros(d); ej[j] = h[j]
        H[j, j] = (objective(x + ej) - 2 * f0 + objective(x - ej)) / h[j] ** 2
        for k in range(j):
            ek = np.zeros(d); ek[k] = h[k]
            H[j, k] = H[k, j] = (objective(x + ej + ek) - objective(x + ej - ek) - objective(x - ej + ek) + objective(x - ej - ek)) / (4 * h[j] * h[k])
    eig = np.linalg.eigvalsh(H)
    condition = float(eig[-1] / eig[0]) if eig[0] > 0 else float('inf')
    if not best.success or boundary or eig[0] <= 1e-6 or condition > 1e9:
        return answer
    cov = np.linalg.inv(H); scores = np.zeros((obs.action.shape[0], d)); J = np.zeros((2, d))
    for j in range(d):
        ej = np.zeros(d); ej[j] = h[j]
        scores[:, j] = (loglik_ext(x + ej, obs, use_sensor, fixed_h, retention_loss) - loglik_ext(x - ej, obs, use_sensor, fixed_h, retention_loss)) / (2 * h[j])
        J[:, j] = (target(expit((x + ej)[:4]), fixed_h) - target(expit((x - ej)[:4]), fixed_h)) / (2 * h[j])
    scores -= scores.mean(axis=0)
    for name, C in (('model', cov), ('sandwich', cov @ (scores.T @ scores) @ cov)):
        v = np.diag(J @ C @ J.T)
        if np.any(v < 0) or not np.all(np.isfinite(v)):
            return answer
        se = np.sqrt(v); answer['ci_' + name] = np.column_stack([est - 1.96 * se, est + 1.96 * se]).tolist()
    answer['interval_valid'] = True
    return answer
