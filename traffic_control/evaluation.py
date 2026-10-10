"""Frozen-policy evaluation, paired comparisons and interpretable examples."""
import numpy as np
import pandas as pd
from .environment import Intersection, SCENARIOS

BASELINES = ('Fixed', 'Actuated', 'QueueGreedy')


def baseline_action(name, env):
    queues = np.array([len(q) for q in env.queues])
    totals = np.array([queues[:2].sum(), queues[2:].sum()])
    if name == 'Fixed':
        return (env.time // 30) % 2
    if name == 'QueueGreedy':
        # Staying wins ties, so an empty intersection does not flicker.
        return env.phase if totals[0] == totals[1] else int(np.argmax(totals))
    if name == 'Actuated':
        other = 1 - env.phase
        if env.green_age >= 60 or (totals[env.phase] == 0 and totals[other] > 0):
            return other
        return env.phase
    raise ValueError(f'Unknown baseline: {name}')


def rollout(controller, scenario, demand_seed, settings, agent=None):
    env = Intersection(settings)
    state = env.reset(scenario, demand_seed)
    done = False
    decisions = []
    while not done:
        if agent is None:
            action = baseline_action(controller, env)
            scores = [float('nan'), float('nan')]
        else:
            actuator = agent['teacher'] if agent['actor'] is None else agent['actor']
            scores = actuator.forward(state)[0]
            action = int(np.argmax(scores))
        decisions.append({'second': env.time, 'action': action,
                          'score_NS': float(scores[0]), 'score_EW': float(scores[1])})
        state, reward, done, metrics = env.step(action)
    return metrics, pd.DataFrame(env.history), pd.DataFrame(decisions)


def evaluate(agents, settings, demand_seeds=range(90000, 90010)):
    rows = []
    training_seeds = sorted(set(seed for _, seed in agents))
    for scenario in SCENARIOS:
        for demand_seed in demand_seeds:
            for baseline in BASELINES:
                metrics, _, _ = rollout(baseline, scenario, demand_seed, settings)
                rows.append({'scenario': scenario, 'algorithm': baseline, 'train_seed': -1,
                             'demand_seed': demand_seed, **metrics})
            for (algorithm, seed), agent in agents.items():
                metrics, _, _ = rollout(algorithm, scenario, demand_seed, settings, agent)
                rows.append({'scenario': scenario, 'algorithm': algorithm, 'train_seed': seed,
                             'demand_seed': demand_seed, **metrics})
    return pd.DataFrame(rows)


def paired_summary(results, bootstrap_seed=482):
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    # Average training seeds within each held-out demand. Bootstrap demand seeds,
    # not every row: outcomes on a shared arrival stream are paired observations.
    grouped = results.groupby(['scenario', 'algorithm', 'demand_seed'])['delay_per_arrival_s'].mean()
    for scenario in SCENARIOS:
        baseline = grouped.loc[(scenario, 'Actuated')]
        for algorithm in results.algorithm.unique():
            current = grouped.loc[(scenario, algorithm)].reindex(baseline.index)
            pairs = np.column_stack([baseline.values, current.values])
            indices = rng.integers(0, len(pairs), size=(2000, len(pairs)))
            resampled = pairs[indices].mean(axis=1)
            improvements = 100. * (resampled[:, 0] - resampled[:, 1]) / resampled[:, 0]
            estimate = 100. * (baseline.mean() - current.mean()) / baseline.mean()
            lower, upper = np.percentile(improvements, [2.5, 97.5])
            rows.append({'scenario': scenario, 'algorithm': algorithm,
                         'delay_s': float(current.mean()), 'improvement_pct': float(estimate),
                         'paired_ci_low_pct': float(lower), 'paired_ci_high_pct': float(upper),
                         'demand_runs': len(pairs)})
    return pd.DataFrame(rows)


def decision_example(actor, scenario, settings, seed=90000, at_second=300):
    env = Intersection(settings)
    state = env.reset(scenario, seed)
    for _ in range(at_second // settings.decision_seconds):
        action = int(np.argmax(actor.forward(state)[0]))
        state, _, done, _ = env.step(action)
    explanation = actor.explain(state)
    explanation.update(scenario=scenario, demand_seed=seed, second=env.time,
                       raw_features=env.raw_features().tolist(),
                       normalized_state=state.tolist(), current_phase=env.phase)
    return explanation


def monotonicity_check(actor, seed=733, count=100):
    rng = np.random.default_rng(seed)
    states = rng.uniform(0., 2., (count, 32))
    states[:, 24:] = np.tile([0, 0, 0, 1, 1, 0, 0, 0], (count, 1))
    derivatives = actor.input_derivatives(states)
    expected = np.sign(actor.parameters['weight'])
    return {'checked_derivatives': int(derivatives.size),
            'sign_violations': int(np.sum(derivatives * expected[None] < -1e-10)),
            'finite': bool(np.all(np.isfinite(derivatives)))}
