"""Checks that the extension reproduces the original model where they overlap, and is correct where it adds."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repository root holds the original history_model.py
import itertools
import unittest
import numpy as np
from scipy.stats import norm
from history_model import Config, Observed, simulate, loglik, fit, pack, start_theta
from history_model_ext import ObservedExt, simulate_ext, loglik_ext, fit_ext


def as_original(obs):
    """Original Observed object: sensor values kept only for the total-mastery channel."""
    z = np.where(obs.zkind == 1, obs.auxiliary, np.nan)
    return Observed(obs.action, obs.response, obs.transition, z, obs.assigned_probability, obs.phase)


class ExtensionChecks(unittest.TestCase):
    def test_likelihood_matches_original(self):
        obs, _ = simulate_ext(Config(n=20, blocks=2), 'recency', seed=3, sensor='total')
        for use, h in ((False, 0.), (True, .8)):
            x = start_theta(use) + np.random.default_rng(1).normal(0, .2, 13 if use else 10)
            np.testing.assert_allclose(loglik_ext(x, obs, use, h), loglik(x, as_original(obs), use, h), rtol=0, atol=1e-10)

    def test_skill_sensor_against_path_enumeration(self):
        x = start_theta(True) + .1
        obs = ObservedExt(np.array([[0, 2, 1, 0]]), np.array([[1, 0, 1, 0]]), np.array([[1, 1, 0, 0]]),
                          np.array([[np.nan, np.nan, .3, 1.1]]), np.array([[0, 0, 2, 3]]),
                          np.zeros((1, 4), int), np.ones((1, 4)))
        mats, p, resp, _, _, _ = pack(x[:10], False, .8, (.15, .15))
        c0, load, sd = x[10], np.exp(x[11]), np.exp(x[12])
        level = {2: lambda s: s % 2, 3: lambda s: s // 2}       # state index: 0=00, 1=10, 2=01, 3=11
        total = 0.
        for u in range(2):
            for states in itertools.product(range(4), repeat=4):
                w = .5 * p[states[0]]
                for t, s in enumerate(states):
                    pr = resp[obs.action[0, t], s]; w *= pr if obs.response[0, t] else 1 - pr
                    if obs.zkind[0, t]:
                        w *= norm.pdf(obs.auxiliary[0, t], c0 + load * level[obs.zkind[0, t]](s), sd)
                    if t < 3:
                        w *= mats[u, obs.action[0, t], s, states[t + 1]] if obs.transition[0, t] == 1 else float(states[t + 1] == s)
                total += w
        self.assertAlmostEqual(loglik_ext(x, obs, True, .8)[0], np.log(total), places=11)

    def test_common_random_numbers_across_designs(self):
        c = Config(n=15, blocks=3)
        runs = [simulate_ext(c, 'recency', 9, sensor, reps) for sensor, reps in (('none', 3), ('total', 3), ('skill', 3), ('none', 6))]
        for obs, truth in runs[1:]:
            np.testing.assert_array_equal(truth['practice_actions'], runs[0][1]['practice_actions'])
            np.testing.assert_array_equal(truth['type'], runs[0][1]['type'])
            np.testing.assert_array_equal(truth['before'][obs.transition == 1].reshape(15, -1),
                                          runs[0][1]['before'][runs[0][0].transition == 1].reshape(15, -1))

    def test_same_learners_across_rules(self):
        a = simulate_ext(Config(n=30), 'recency', 4, 'none')[1]; b = simulate_ext(Config(n=30), 'balanced', 4, 'none')[1]
        np.testing.assert_array_equal(a['type'], b['type'])
        np.testing.assert_array_equal(a['before'][:, 0], b['before'][:, 0])

    def test_yoked_replays_partner(self):
        c = Config(n=25)
        partner = simulate_ext(c, 'recency', 5, 'none')[1]['practice_actions']
        obs, truth = simulate_ext(c, 'recency', 6, 'none', yoked=partner)
        np.testing.assert_array_equal(truth['practice_actions'], partner)
        self.assertTrue((obs.assigned_probability[obs.transition == 1] == 1).all())

    def test_simulator_matches_original_in_distribution(self):
        c = Config(n=1500)
        orig = simulate(c, 'recency', 11)[0]; ext = simulate_ext(c, 'recency', 11, 'total')[0]
        for phase in (1, 4, 6):
            m0 = orig.response[orig.phase == phase].mean(); m1 = ext.response[ext.phase == phase].mean()
            self.assertLess(abs(m0 - m1), .02, f'phase {phase}: {m0:.3f} vs {m1:.3f}')
        self.assertLess(abs(np.nanmean(orig.auxiliary) - np.nanmean(ext.auxiliary)), .05)

    def test_fit_matches_original(self):
        obs, _ = simulate_ext(Config(n=60), 'recency', 21, 'total')
        for use in (False, True):
            r0 = fit(as_original(obs), auxiliary=use, fixed_h=.8, seed=2, starts=2)
            r1 = fit_ext(obs, use_sensor=use, fixed_h=.8, seed=2, starts=2)
            np.testing.assert_allclose(r1['estimate'], r0['estimate'], atol=1e-9)
            np.testing.assert_allclose(r1['ci_model'], r0['ci_model'], atol=1e-9)
            np.testing.assert_allclose(r1['ci_sandwich'], r0['ci_sandwich'], atol=1e-9)


    def test_practice_sensor(self):
        c = Config(n=10, blocks=2)
        base = simulate_ext(c, 'recency', 8, 'none')[1]
        for kind in ('total', 'skill'):
            obs, truth = simulate_ext(c, 'recency', 8, kind, sensor_at='practice')
            np.testing.assert_array_equal(truth['practice_actions'], base['practice_actions'])      # same learners and practice
            self.assertTrue((obs.zkind[obs.transition == 1] > 0).all() and (obs.zkind[obs.transition != 1] == 0).all())
            if kind == 'skill':
                np.testing.assert_array_equal(obs.zkind[obs.transition == 1], np.array([2, 3, 1])[obs.action[obs.transition == 1]])


    def test_no_scheduled_exploration(self):
        c = Config(n=25, epsilon=0.)
        base_obs, base = simulate_ext(c, 'recency', 12, 'none')
        obs, truth = simulate_ext(c, 'recency', 12, 'none', scheduled_exploration=False)
        self.assertEqual((base_obs.phase == 2).sum(), 25 * 3 * c.blocks)              # the default design schedules 3 per block
        self.assertEqual((obs.phase == 2).sum(), 0)                                    # none remain
        self.assertEqual((obs.phase == 3).sum(), 25 * 12 * c.blocks)                   # every block trial is adaptive
        np.testing.assert_array_equal(truth['type'], base['type'])                     # same learners
        np.testing.assert_array_equal(truth['practice_actions'][:, :18], base['practice_actions'][:, :18])   # same initial practice

    def test_practice_sparse_sensor(self):
        c = Config(n=15)
        base = simulate_ext(c, 'recency', 13, 'none')
        segments = [(0, 18)] + [(18 + 12 * b, 30 + 12 * b) for b in range(c.blocks)]
        for kind, per_segment in (('total', 1), ('skill', 2)):
            obs, truth = simulate_ext(c, 'recency', 13, kind, sensor_at='practice_sparse')
            np.testing.assert_array_equal(truth['practice_actions'], base[1]['practice_actions'])   # same learners and practice
            np.testing.assert_array_equal(truth['before'], base[1]['before'])                        # same latent trajectories
            probes_obs = simulate_ext(c, 'recency', 13, kind, sensor_at='probes')[0]
            self.assertEqual((obs.zkind > 0).sum(), (probes_obs.zkind > 0).sum())                     # equal number of readings
            for i in range(c.n):
                prac = np.where(obs.transition[i] == 1)[0]
                read = np.where(obs.zkind[i] > 0)[0]
                self.assertTrue(np.isin(read, prac).all())                                           # all on practice trials
                k = np.searchsorted(prac, read)
                for lo, hi in segments:
                    self.assertEqual(((k >= lo) & (k < hi)).sum(), per_segment)
                if kind == 'skill':
                    np.testing.assert_array_equal(obs.zkind[i, read], np.array([2, 3, 1])[obs.action[i, read]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
