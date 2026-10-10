"""Inspect a saved DRHQ decision without retraining anything.

Run from the repository root:
    python examples/inspect_decision.py --scenario tidal --second 900
"""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from traffic_control.config import Settings, ACTION_NAMES, DIRECTIONS, FEATURE_NAMES
from traffic_control.training import load_agent
from traffic_control.evaluation import decision_example


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',choices=['balanced','tidal'],default='tidal')
    parser.add_argument('--second',type=int,default=900)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--results',default='results')
    args=parser.parse_args()
    folder=Path(args.results)
    manifest=json.loads((folder/'manifest.json').read_text())
    settings=Settings(**manifest['settings'])
    if not 0<=args.second<settings.horizon:
        parser.error('second must be inside the episode horizon')
    agent=load_agent(folder/'models'/f'DRHQ_{args.seed}.npz','DRHQ',settings)
    example=decision_example(agent['actor'],args.scenario,settings,at_second=args.second)
    print(pd.DataFrame(example['raw_features'],index=DIRECTIONS,columns=FEATURE_NAMES).round(3))
    terms=np.asarray(example['contributions']).sum(axis=1)
    frame=pd.DataFrame(terms,index=ACTION_NAMES,columns=FEATURE_NAMES)
    frame['score']=example['scores']
    print('\nScore terms:\n',frame.round(4))
    np.testing.assert_allclose(terms.sum(axis=1),example['scores'])
    print('\nChosen:',ACTION_NAMES[example['action']])

if __name__=='__main__': main()
