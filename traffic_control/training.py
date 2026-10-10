"""Online DQN teacher training and the three regulatable actuators."""
import numpy as np
from .environment import Intersection, SCENARIOS
from .networks import QNetwork, RegulatablePolicy

ALGORITHMS = ('DQN', 'DRQ', 'DRSQ', 'DRHQ')


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.states = np.zeros((capacity, 32))
        self.next_states = np.zeros((capacity, 32))
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity)
        self.done = np.zeros(capacity)

    def append(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.done[i] = action, reward, done
        self.position = (i + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size, rng):
        indices = rng.choice(self.size, batch_size, replace=False)
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.done[indices])


def td_targets(rewards, done, next_scores, gamma):
    return rewards + gamma * (1. - done) * np.max(next_scores, axis=1)


def train_agent(algorithm, seed, settings, verbose=True):
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    settings.validate()
    rng = np.random.default_rng(seed)
    teacher = QNetwork(seed, settings.hidden, settings.learning_rate)
    target = teacher.clone()
    actor = None if algorithm == 'DQN' else RegulatablePolicy(settings.policy_learning_rate)
    replay = Replay(settings.replay_capacity)
    env = Intersection(settings)
    log = []
    updates = 0
    steps = 0
    for episode in range(settings.episodes):
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        # Evaluation uses a separate seed range and never tunes parameters.
        demand_seed = 10000 + seed * 1000 + episode
        state = env.reset(scenario, demand_seed)
        fraction = episode / max(settings.episodes - 1, 1)
        epsilon = settings.epsilon_start + fraction * (settings.epsilon_end - settings.epsilon_start)
        q_losses, g_losses, agreements = [], [], []
        done = False
        while not done:
            actuator = teacher if actor is None else actor
            action = int(rng.integers(2)) if rng.random() < epsilon else int(np.argmax(actuator.forward(state)[0]))
            next_state, reward, done, info = env.step(action)
            replay.append(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            if replay.size < settings.warmup:
                continue
            batch = replay.sample(settings.batch_size, rng)
            s, a, r, ns, terminal = batch
            y = td_targets(r, terminal, target.forward(ns), settings.gamma)
            q_losses.append(teacher.train(s, a, y))
            updates += 1
            if updates % settings.target_interval == 0:
                target.sync(teacher)
            if actor is not None:
                for _ in range(settings.policy_updates):
                    s, a, r, ns, terminal = replay.sample(settings.batch_size, rng)
                    y = td_targets(r, terminal, target.forward(ns), settings.gamma)
                    q_scores = teacher.forward(s)
                    g_losses.append(actor.train(s, q_scores, algorithm, a, y))
                    agreements.append(np.mean(np.argmax(actor.forward(s), axis=1) == np.argmax(q_scores, axis=1)))
        log.append({'algorithm': algorithm, 'train_seed': seed, 'episode': episode + 1,
                    'scenario': scenario, 'epsilon': epsilon, 'q_loss': float(np.mean(q_losses)) if q_losses else 0.,
                    'g_loss': float(np.mean(g_losses)) if g_losses else 0.,
                    'replay_agreement': float(np.mean(agreements)) if agreements else 1.,
                    **info})
        if verbose and (episode + 1) % 10 == 0:
            print(f'{algorithm:4s} seed={seed} episode={episode+1:3d} delay={info["delay_per_arrival_s"]:6.2f}s', flush=True)
    return {'teacher': teacher, 'actor': actor, 'log': log, 'seed': seed,
            'algorithm': algorithm, 'updates': updates, 'steps': steps}


def save_agent(agent, path):
    payload = {f'q_{k}': v for k, v in agent['teacher'].parameters.items()}
    if agent['actor'] is not None:
        payload.update({f'g_{k}': v for k, v in agent['actor'].parameters.items()})
    np.savez_compressed(path, **payload)


def load_agent(path, algorithm, settings):
    teacher = QNetwork(hidden=settings.hidden)
    actor = None if algorithm == 'DQN' else RegulatablePolicy()
    with np.load(path) as values:
        for key in teacher.parameters:
            teacher.parameters[key][:] = values[f'q_{key}']
        if actor is not None:
            for key in actor.parameters:
                actor.parameters[key][:] = values[f'g_{key}']
    return {'teacher': teacher, 'actor': actor, 'algorithm': algorithm}
