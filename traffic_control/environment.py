"""A small FIFO queue simulator with enforced signal clearance.

This is a teaching environment, not a SUMO replacement. Opposing through
movements share green. Arrivals enter stopped queues immediately. No turning,
acceleration, spillback, pedestrians or sensor error is modeled.
"""
from collections import deque
import numpy as np
from .config import Settings, MOVEMENTS, SCALES

SCENARIOS = ('balanced', 'tidal')


def demand_rates(scenario, horizon):
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown demand scenario: {scenario}')
    rates = np.full((horizon, 4), 0.30)
    if scenario == 'tidal':
        midpoint = horizon // 2
        rates[:midpoint] = [0.54, 0.48, 0.12, 0.16]
        rates[midpoint:] = [0.12, 0.16, 0.54, 0.48]
    return rates


def make_arrivals(scenario, horizon, seed):
    # Generate demand before control starts, so every controller sees the same traffic.
    return np.random.default_rng(seed).poisson(demand_rates(scenario, horizon))


class Intersection:
    observation_size = 32
    action_count = 2

    def __init__(self, settings=None):
        self.settings = (settings or Settings()).validate()

    def reset(self, scenario='balanced', seed=0, arrivals=None):
        self.arrivals = make_arrivals(scenario, self.settings.horizon, seed) if arrivals is None else np.asarray(arrivals)
        if self.arrivals.shape != (self.settings.horizon, 4):
            raise ValueError('arrivals must have shape (horizon, 4)')
        if np.any(self.arrivals < 0) or np.any(self.arrivals != self.arrivals.astype(int)):
            raise ValueError('arrival counts must be nonnegative integers')
        self.queues = [deque() for _ in range(4)]
        self.time = 0
        self.phase = 0
        self.green_age = 0
        self.delay = 0
        self.arrived = 0
        self.departed = 0
        self.switches = 0
        self.departure_waits = []
        self.history = []
        return self.observe()

    def raw_features(self):
        values = np.zeros((4, 6), dtype=float)
        for lane, queue in enumerate(self.queues):
            count = len(queue)
            total_wait = sum(self.time - entry for entry in queue)
            # One second of scheduled arrivals is a synthetic approach detector.
            approaching = self.arrivals[self.time, lane] if self.time < len(self.arrivals) else 0
            values[lane] = [count, approaching, total_wait,
                            total_wait / count if count else 0, count,
                            12.0 if approaching else 0.0]
        return values

    def observe(self):
        traffic = (self.raw_features() / np.asarray(SCALES)).ravel()
        flags = np.zeros((2, 4), dtype=float)
        flags[:, 0] = 1.0  # full clearance when changing direction
        flags[self.phase] = [0., 0., 0., 1.]  # no clearance for the current green
        return np.concatenate([traffic, flags.ravel()])

    def step(self, action):
        if action not in (0, 1):
            raise ValueError('action must be 0 or 1')
        if self.time >= self.settings.horizon:
            raise RuntimeError('episode has ended; call reset before step')
        switch = int(action != self.phase)
        if switch:
            self.switches += 1
            self.green_age = 0
        clearance = switch * (self.settings.yellow_seconds + self.settings.all_red_seconds)
        start_delay = self.delay
        self.phase = int(action)
        for tick in range(self.settings.decision_seconds):
            for lane, count in enumerate(self.arrivals[self.time]):
                for _ in range(int(count)):
                    self.queues[lane].append(self.time)
                self.arrived += int(count)
            if tick < clearance:
                signal = 'yellow' if tick < self.settings.yellow_seconds else 'all_red'
                served = ()
            else:
                signal = 'green'
                served = MOVEMENTS[action]
                self.green_age += 1
            for lane in served:
                for _ in range(min(len(self.queues[lane]), self.settings.service_per_second)):
                    entry = self.queues[lane].popleft()
                    self.departure_waits.append(self.time - entry)
                    self.departed += 1
            counts = [len(q) for q in self.queues]
            # Every remaining vehicle waits through this one-second interval.
            self.delay += sum(counts)
            self.history.append({'second': self.time, 'phase': self.phase,
                                 'signal': signal, **{f'q{i}': n for i, n in enumerate(counts)},
                                 'arrived': self.arrived, 'departed': self.departed,
                                 'delay_vehicle_seconds': self.delay})
            self.time += 1
        done = self.time >= self.settings.horizon
        reward = -(self.delay - start_delay) / 100.0
        return self.observe(), reward, done, self.metrics()

    def metrics(self):
        pending = sum(len(q) for q in self.queues)
        waits = np.asarray(self.departure_waits)
        return {'delay_per_arrival_s': self.delay / max(self.arrived, 1),
                'delay_vehicle_seconds': self.delay, 'arrived': self.arrived,
                'departed': self.departed, 'pending': pending,
                'throughput_fraction': self.departed / max(self.arrived, 1),
                'completed_mean_wait_s': float(waits.mean()) if len(waits) else 0.,
                'completed_p95_wait_s': float(np.percentile(waits, 95)) if len(waits) else 0.,
                'switches': self.switches,
                'conservation_ok': self.arrived == self.departed + pending}
