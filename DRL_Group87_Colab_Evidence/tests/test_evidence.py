"""Checks that distinguish a frozen evaluation from continued training."""
import unittest
import numpy as np
from traffic_control.config import Settings
from traffic_control.environment import Intersection
from traffic_control.networks import QNetwork, RegulatablePolicy
from traffic_control.proof import run_diagnostics


class EvidenceTests(unittest.TestCase):
    def test_stress_evaluation_is_frozen_and_conserves_vehicles(self):
        agent = {'teacher': QNetwork(seed=19), 'actor': RegulatablePolicy()}
        results, frozen = run_diagnostics(agent)
        self.assertTrue(frozen)
        self.assertEqual(len(results), 16)
        self.assertTrue(results.conservation_ok.all())
        self.assertTrue((results[results.case == 'overloaded'].pending > 0).all())
        self.assertTrue((results[results.case == 'empty'].delay_vehicle_seconds == 0).all())

    def test_unfinished_vehicles_contribute_to_delay(self):
        env = Intersection(Settings(horizon=20))
        arrivals = np.zeros((20, 4), dtype=int)
        arrivals[0, 2] = 3
        env.reset(arrivals=arrivals)
        env.step(0)
        _, _, _, metrics = env.step(0)
        self.assertEqual(metrics['departed'], 0)
        self.assertEqual(metrics['pending'], 3)
        self.assertEqual(metrics['delay_vehicle_seconds'], 60)
        self.assertEqual(metrics['delay_per_arrival_s'], 20)


if __name__ == '__main__':
    unittest.main()
