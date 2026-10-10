"""Check mechanics and derivatives before trusting experiment plots."""
import unittest
import numpy as np
from traffic_control.config import Settings
from traffic_control.environment import Intersection, make_arrivals
from traffic_control.networks import QNetwork, RegulatablePolicy
from traffic_control.training import Replay, td_targets, train_agent, save_agent, load_agent
from traffic_control.evaluation import monotonicity_check
import tempfile
from pathlib import Path


class EnvironmentTests(unittest.TestCase):
    def test_conservation_and_delay_identity(self):
        cfg = Settings(horizon=100)
        env = Intersection(cfg)
        env.reset('tidal', 12)
        for action in [0, 1] * 5:
            _, _, _, metrics = env.step(action)
            self.assertTrue(metrics['conservation_ok'])
        recorded = sum(sum(row[f'q{i}'] for i in range(4)) for row in env.history)
        self.assertEqual(recorded, metrics['delay_vehicle_seconds'])
        pending_wait = sum(env.time - entry for q in env.queues for entry in q)
        self.assertEqual(sum(env.departure_waits) + pending_wait, env.delay)

    def test_clearance_has_no_departures(self):
        cfg = Settings(horizon=20)
        env = Intersection(cfg)
        arrivals = np.zeros((20, 4), dtype=int)
        arrivals[0, 2] = 5
        env.reset(arrivals=arrivals)
        env.step(1)
        self.assertEqual([r['signal'] for r in env.history[:3]], ['yellow', 'yellow', 'all_red'])
        self.assertTrue(all(r['departed'] == 0 for r in env.history[:3]))
        self.assertEqual(env.history[3]['departed'], 1)

    def test_empty_traffic_has_zero_delay(self):
        env = Intersection(Settings(horizon=20))
        env.reset(arrivals=np.zeros((20, 4), dtype=int))
        env.step(1)
        _, reward, done, metrics = env.step(0)
        self.assertEqual(metrics['delay_vehicle_seconds'], 0)
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError): env.step(0)

    def test_seeded_traffic(self):
        np.testing.assert_array_equal(make_arrivals('tidal', 100, 7), make_arrivals('tidal', 100, 7))
        self.assertFalse(np.array_equal(make_arrivals('tidal', 100, 7), make_arrivals('tidal', 100, 8)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): Settings(horizon=11).validate()
        with self.assertRaises(ValueError): Settings(yellow_seconds=10).validate()
        env = Intersection(Settings(horizon=20))
        with self.assertRaises(ValueError): env.reset(arrivals=np.full((20, 4), -1))
        env.reset()
        with self.assertRaises(ValueError): env.step(3)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(16)
        self.states = self.rng.uniform(.1, 1., (5, 32))
        self.states[:, 24:] = np.tile([0,0,0,1,1,0,0,0], (5,1))
        self.actions = np.array([0,1,0,1,0])
        self.targets = self.rng.normal(size=5)
        self.teacher = self.rng.normal(size=(5,2))

    def check_gradient(self, model, objective, gradients):
        for key, values in model.parameters.items():
            positions = [tuple(0 for _ in values.shape), tuple(n-1 for n in values.shape)]
            for index in positions:
                original = values[index]
                step = 1e-5
                values[index] = original + step
                high = objective()
                values[index] = original - step
                low = objective()
                values[index] = original
                numerical = (high-low) / (2*step)
                self.assertAlmostEqual(numerical, gradients[key][index], places=5, msg=f'{key}{index}')

    def test_q_backpropagation(self):
        q = QNetwork(seed=4)
        loss, gradients = q.gradients(self.states, self.actions, self.targets)
        self.check_gradient(q, lambda:q.gradients(self.states,self.actions,self.targets)[0], gradients)

    def test_policy_gradients_all_variants(self):
        for variant in ('DRQ','DRSQ','DRHQ'):
            with self.subTest(variant=variant):
                actor = RegulatablePolicy()
                def objective():
                    return actor.loss_and_gradients(self.states,self.teacher,variant,self.actions,self.targets)[0]
                _, gradients = actor.loss_and_gradients(self.states,self.teacher,variant,self.actions,self.targets)
                self.check_gradient(actor, objective, gradients)

    def test_terminal_target_no_bootstrap(self):
        target = td_targets(np.array([2.,3.]), np.array([1.,0.]),np.array([[99.,20.],[4.,5.]]), .9)
        np.testing.assert_allclose(target,[2.,7.5])

    def test_monotonicity_and_explanation(self):
        actor = RegulatablePolicy()
        actor.parameters['weight'][0,0,0] = -2.
        result = monotonicity_check(actor)
        self.assertTrue(result['finite'])
        self.assertEqual(result['sign_violations'],0)
        explanation = actor.explain(self.states[0])
        np.testing.assert_allclose(np.array(explanation['contributions']).sum(axis=(1,2)),explanation['scores'])
        self.assertTrue(np.all(np.isfinite(actor.input_derivatives(np.zeros((1,32))))))

    def test_replay_wrap(self):
        replay = Replay(3)
        for i in range(8): replay.append(np.ones(32)*i,0,i,np.ones(32),False)
        self.assertEqual(replay.size,3)
        s,a,r,ns,d = replay.sample(3,self.rng)
        self.assertEqual(set(r),{5.,6.,7.})

    def test_training_and_checkpoint_roundtrip(self):
        cfg = Settings(horizon=200,episodes=2,warmup=32)
        for algorithm in ('DQN','DRQ','DRSQ','DRHQ'):
            agent = train_agent(algorithm,3,cfg,verbose=False)
            self.assertGreater(agent['updates'],0)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.npz'
                save_agent(agent,path)
                loaded = load_agent(path,algorithm,cfg)
                np.testing.assert_allclose(agent['teacher'].forward(self.states),loaded['teacher'].forward(self.states))
                if agent['actor']:
                    np.testing.assert_allclose(agent['actor'].forward(self.states),loaded['actor'].forward(self.states))

if __name__ == '__main__': unittest.main()
