"""Shared settings. Times are in seconds and rates are vehicles/second."""
from dataclasses import dataclass, asdict

DIRECTIONS = ('North', 'South', 'East', 'West')
ACTION_NAMES = ('North / South', 'East / West')
MOVEMENTS = ((0, 1), (2, 3))
FEATURE_NAMES = ('queued', 'approaching', 'total_wait', 'mean_wait', 'queue_per_lane', 'speed')
SCALES = (30., 5., 1800., 120., 30., 15.)

@dataclass
class Settings:
    horizon: int = 1200
    decision_seconds: int = 10
    yellow_seconds: int = 2
    all_red_seconds: int = 1
    service_per_second: int = 1
    episodes: int = 50
    batch_size: int = 32
    replay_capacity: int = 12000
    target_interval: int = 100
    warmup: int = 128
    gamma: float = 0.95
    learning_rate: float = 0.001
    policy_learning_rate: float = 0.003
    epsilon_start: float = 0.30
    epsilon_end: float = 0.03
    policy_updates: int = 2
    hidden: int = 32

    def validate(self):
        if self.horizon < self.decision_seconds or self.horizon % self.decision_seconds:
            raise ValueError('horizon must be a positive multiple of decision_seconds')
        if self.yellow_seconds < 0 or self.all_red_seconds < 0:
            raise ValueError('clearance durations cannot be negative')
        if self.yellow_seconds + self.all_red_seconds >= self.decision_seconds:
            raise ValueError('each decision must leave some green time')
        if self.service_per_second < 1 or self.episodes < 1 or self.batch_size < 1:
            raise ValueError('service, episodes and batch_size must be positive')
        if self.replay_capacity < self.batch_size or self.warmup < self.batch_size:
            raise ValueError('replay capacity and warmup must cover a minibatch')
        if not 0 <= self.gamma < 1 or self.target_interval < 1:
            raise ValueError('invalid discount or target interval')
        return self

    def to_dict(self):
        return asdict(self)
