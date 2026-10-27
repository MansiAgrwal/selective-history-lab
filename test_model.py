"""Executable checks of the statistical implementation, not scientific validation."""
import itertools
import unittest
import numpy as np
from scipy.stats import norm
from history_model import (Config, Observed, simulate, policy, retained_indices,
    transition_matrix, initial_dist, target, loglik, pack, start_theta, score_heldout)

class ModelChecks(unittest.TestCase):
    def test_transition_mass_and_forbidden_gain(self):
        for e in range(3):
            p = transition_matrix((.08,.045,.055,.04),(.008,.008),e)
            np.testing.assert_allclose(p.sum(1),1)
            self.assertTrue((p>=0).all())
        self.assertEqual(transition_matrix((.1,)*4,(0,0),0)[0,2],0)

    def test_memory_and_exploration(self):
        h=[(0,1)]*20+[(1,0)]*5+[(2,0)]*4
        self.assertEqual(retained_indices(h,'recency'),list(range(11,29)))
        k=retained_indices(h,'balanced')
        self.assertEqual(len(k),18)
        self.assertEqual(len(set(k)),18)
        for rule in ('balanced','recency'):
            p=policy(h,rule)
            self.assertAlmostEqual(p.sum(),1)
            self.assertGreaterEqual(p.min(),.1/3)

    def test_continuity_schedule_and_no_probe_learning(self):
        obs,truth=simulate(Config(n=6,blocks=2),seed=17)
        np.testing.assert_array_equal(truth['before'][:,1:],truth['after'][:,:-1])
        m=obs.transition==0
        np.testing.assert_array_equal(truth['before'][m],truth['after'][m])
        self.assertTrue((np.sum(obs.phase==2,axis=1)==6).all())
        self.assertTrue((np.sum(obs.phase==3,axis=1)==18).all())
        self.assertTrue((np.sum(np.isfinite(obs.auxiliary),axis=1)==3).all())
        self.assertTrue((obs.assigned_probability[obs.phase==3]>=.1/3).all())

    def test_forward_against_explicit_path_sum(self):
        # Enumerate all state paths and a persistent, rather than renewed, type.
        x=start_theta(True)
        obs=Observed(np.array([[0,2,1]]),np.array([[1,0,1]]),
            np.array([[1,1,0]]),np.array([[.4,np.nan,1.2]]),
            np.ones((1,3)),np.zeros((1,3),dtype=int))
        mats,p,resp,ret,mu,sd=pack(x,True,.8,(.15,.15))
        total=0.
        for u in range(2):
            for states in itertools.product(range(4),repeat=3):
                w=.5*p[states[0]]
                for t,s in enumerate(states):
                    pr=resp[obs.action[0,t],s]
                    w*=pr if obs.response[0,t] else 1-pr
                    if np.isfinite(obs.auxiliary[0,t]):
                        w*=norm.pdf(obs.auxiliary[0,t],mu[s],sd)
                    if t<2:w*=mats[u,obs.action[0,t],s,states[t+1]]
                total+=w
        self.assertAlmostEqual(loglik(x,obs,True,.8)[0],np.log(total),places=11)

    def test_sensor_is_not_required_for_behavior_likelihood(self):
        a,_=simulate(Config(n=3,blocks=1),seed=3)
        b=Observed(a.action,a.response,a.transition,a.auxiliary*999,
                   a.assigned_probability,a.phase)
        np.testing.assert_equal(loglik(start_theta(),a),loglik(start_theta(),b))

    def test_current_auxiliary_does_not_leak_into_prediction(self):
        a=Observed(np.array([[0]]),np.array([[1]]),np.array([[0]]),
                   np.array([[0.]]),np.ones((1,1)),np.zeros((1,1),int))
        b=Observed(a.action,a.response,a.transition,np.array([[10.]]),
                   a.assigned_probability,a.phase)
        self.assertEqual(score_heldout(start_theta(True),a,0,True),
                         score_heldout(start_theta(True),b,0,True))

    def test_adaptive_counts_are_not_fixed_counts(self):
        # Gain .2; A first, then choose A with .9 if mastered and .1 otherwise.
        # Conditioning on AA is selection on a post-first-action state.
        selected_no_mastery=(.8*.1*.8)/(.2*.9+.8*.1)
        fixed_no_mastery=.8**2
        self.assertAlmostEqual(selected_no_mastery,.2461538461538462)
        self.assertAlmostEqual(fixed_no_mastery,.64)
        self.assertNotAlmostEqual(selected_no_mastery,fixed_no_mastery)

    def test_standardized_target(self):
        np.testing.assert_allclose(target((.08,.045,.055,.04),0),[-.035,-.015])
        self.assertEqual(target((.08,.045,.055,.04),.8).shape,(2,))

if __name__=='__main__':unittest.main(verbosity=2)
