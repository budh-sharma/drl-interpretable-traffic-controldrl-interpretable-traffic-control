"""Run the study and write its raw evidence. No numbers are typed into plots."""
import argparse
from pathlib import Path
import json
import time
import hashlib
import platform
import sys
import numpy as np
import pandas as pd
from traffic_control.config import Settings
from traffic_control.training import ALGORITHMS, train_agent, save_agent
from traffic_control.evaluation import evaluate, paired_summary, rollout, decision_example, monotonicity_check


def run_study(output='results', episodes=50, seeds=(11,22,33), evaluation_runs=10, horizon=1200):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'models').mkdir(exist_ok=True)
    cfg = Settings(episodes=episodes,horizon=horizon)
    agents = {}
    logs = []
    for algorithm in ALGORITHMS:
        for seed in seeds:
            agent = train_agent(algorithm,seed,cfg)
            agents[(algorithm,seed)] = agent
            logs.extend(agent['log'])
            save_agent(agent,output/'models'/f'{algorithm}_{seed}.npz')
    training = pd.DataFrame(logs)
    training.to_csv(output/'training.csv',index=False)
    results = evaluate(agents,cfg,range(90000,90000+evaluation_runs))
    results.to_csv(output/'evaluation.csv',index=False)
    summary = paired_summary(results)
    summary.to_csv(output/'summary.csv',index=False)
    results[results.train_seed>=0].groupby(['scenario','algorithm','train_seed']).delay_per_arrival_s.mean().reset_index().to_csv(output/'training_seed_summary.csv',index=False)
    # Always use the first declared training seed for demonstrations, never the best.
    example_agent = agents[('DRHQ',seeds[0])]
    examples = [decision_example(example_agent['actor'],'balanced',cfg,at_second=horizon//4),
                decision_example(example_agent['actor'],'tidal',cfg,at_second=3*horizon//4)]
    (output/'examples.json').write_text(json.dumps(examples,indent=2))
    for scenario in ('balanced','tidal'):
        for controller in ('Fixed','Actuated','QueueGreedy','DRHQ'):
            metrics,history,decisions = rollout(controller,scenario,90000,cfg,
                                              example_agent if controller=='DRHQ' else None)
            history.to_csv(output/f'trace_{scenario}_{controller}.csv',index=False)
            decisions.to_csv(output/f'decisions_{scenario}_{controller}.csv',index=False)
    checks = {str(seed): monotonicity_check(agents[('DRHQ',seed)]['actor']) for seed in seeds}
    (output/'monotonicity.json').write_text(json.dumps(checks,indent=2))
    manifest = {'scope':'Synthetic queue simulator; algorithm demonstration, not SUMO replication',
                'settings':cfg.to_dict(),'training_seeds':list(seeds),
                'evaluation_seeds':list(range(90000,90000+evaluation_runs)),
                'algorithms':list(ALGORITHMS),'baseline_names':['Fixed','Actuated','QueueGreedy'],
                'demonstration_seed':seeds[0], 'python':sys.version, 'numpy':np.__version__,
                'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-start,
                'training_episodes_total':len(training), 'evaluation_rollouts':len(results),
                'bootstrap_unit':'held-out demand stream after averaging training seeds',
                'files_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                for name in ['training.csv','evaluation.csv','summary.csv','examples.json']}}
    manifest['source_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path('traffic_control').glob('*.py'))}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from traffic_control.visuals import make_figures
    make_figures(output)
    print('\nFrozen evaluation (seconds of in-horizon queue delay per arrival):')
    print(summary.round(2).to_string(index=False))
    print(f'Artifacts written to {output.resolve()} in {manifest["elapsed_seconds"]:.1f}s',flush=True)
    return agents, summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results')
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    parser.add_argument('--evaluation-runs',type=int,default=10)
    parser.add_argument('--horizon',type=int,default=1200)
    args=parser.parse_args()
    run_study(args.output,args.episodes,tuple(args.seeds),args.evaluation_runs,args.horizon)
