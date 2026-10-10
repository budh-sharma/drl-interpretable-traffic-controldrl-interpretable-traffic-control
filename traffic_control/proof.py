"""Small diagnostic runs for the presentation's proof of work."""
from pathlib import Path
import json
import platform
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import Settings
from .environment import Intersection
from .evaluation import baseline_action
from .training import load_agent


def parameter_copy(agent):
    return {name: {key: value.copy() for key, value in model.parameters.items()}
            for name, model in agent.items()
            if name in ('teacher', 'actor') and model is not None}


def unchanged_parameters(agent, before):
    return all(np.array_equal(value, agent[name].parameters[key])
               for name, parameters in before.items() for key, value in parameters.items())


def diagnostic_arrivals(horizon):
    empty = np.zeros((horizon, 4), dtype=int)
    one_sided = empty.copy()
    one_sided[0, 2] = 30
    reversal = empty.copy()
    reversal[0, :2] = 20
    reversal[horizon // 2, 2:] = 20
    # Deliberately above service capacity. A growing queue is expected here.
    overloaded = np.full((horizon, 4), 2, dtype=int)
    return {'empty': empty, 'one_sided_burst': one_sided,
            'burst_reversal': reversal, 'overloaded': overloaded}


def run_diagnostics(actor_agent, settings=None):
    cfg = settings or Settings(horizon=200)
    before = parameter_copy(actor_agent)
    rows = []
    for case, arrivals in diagnostic_arrivals(cfg.horizon).items():
        for controller in ('Fixed', 'Actuated', 'QueueGreedy', 'DRHQ'):
            env = Intersection(cfg)
            state = env.reset(arrivals=arrivals)
            done = False
            while not done:
                if controller == 'DRHQ':
                    scores = actor_agent['actor'].forward(state)[0]
                    assert np.all(np.isfinite(scores)), f'{case}: nonfinite scores'
                    action = int(np.argmax(scores))
                else:
                    action = baseline_action(controller, env)
                state, _, done, metrics = env.step(action)
            assert metrics['conservation_ok'], f'{case}: vehicle accounting failed'
            assert all(np.isfinite(v) for v in metrics.values()), f'{case}: nonfinite metrics'
            pending_wait = sum(env.time - entry for queue in env.queues for entry in queue)
            assert np.isclose(sum(env.departure_waits) + pending_wait, env.delay)
            if case == 'empty':
                assert metrics['delay_vehicle_seconds'] == 0 and metrics['pending'] == 0
            rows.append({'case': case, 'algorithm': controller, **metrics})
    frozen = unchanged_parameters(actor_agent, before)
    assert frozen, 'Evaluation changed learned parameters'
    return pd.DataFrame(rows), frozen


def save_proof(folder='results', contributor='', contribution=''):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    cfg = Settings(**manifest['settings'])
    seed = manifest['demonstration_seed']
    agent = load_agent(folder / 'models' / f'DRHQ_{seed}.npz', 'DRHQ', cfg)
    diagnostics, frozen = run_diagnostics(agent)
    diagnostics.to_csv(folder / 'diagnostics.csv', index=False)
    evaluation = pd.read_csv(folder / 'evaluation.csv')
    tests = (folder / 'tests.txt').read_text() if (folder / 'tests.txt').exists() else ''
    checks = {
        'training_episodes': manifest['training_episodes_total'],
        'frozen_evaluation_rollouts': len(evaluation),
        'diagnostic_rollouts': len(diagnostics),
        'conservation_all_rollouts': bool((evaluation.arrived == evaluation.departed + evaluation.pending).all()),
        'diagnostic_parameters_unchanged': bool(frozen),
        'tests_reported_ok': '\nOK' in tests,
        'metric': 'waiting inside horizon / all arrivals, seconds',
        'scope': manifest['scope'],
    }
    (folder / 'proof_summary.json').write_text(json.dumps(checks, indent=2))
    record = {
        'contributor_name': contributor.strip(),
        'actual_contribution': contribution.strip(),
        'status': 'recorded' if contributor.strip() and contribution.strip() else 'pending personal details',
        'utc_time': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'platform': platform.platform(), 'checks': checks,
        'result_hashes': manifest['files_sha256'],
    }
    (folder / 'contributor_run.json').write_text(json.dumps(record, indent=2))
    return diagnostics, checks, record
