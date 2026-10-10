"""Evaluate saved control on a user-specified constant arrival profile.

The demand is outside the declared experiment if its rates differ. Its
outcome must be reported separately from the presentation's two scenarios.
"""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from traffic_control.config import Settings
from traffic_control.environment import Intersection
from traffic_control.training import load_agent
from traffic_control.evaluation import baseline_action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rates',nargs=4,type=float,default=[.30,.30,.30,.30],metavar=('N','S','E','W'))
    parser.add_argument('--traffic-seed',type=int,default=92000)
    parser.add_argument('--results',default='results')
    args=parser.parse_args()
    if any(rate<0 for rate in args.rates):parser.error('rates cannot be negative')
    folder=Path(args.results)
    manifest=json.loads((folder/'manifest.json').read_text())
    settings=Settings(**manifest['settings'])
    arrivals=np.random.default_rng(args.traffic_seed).poisson(np.tile(args.rates,(settings.horizon,1)))
    agent=load_agent(folder/'models'/f'DRHQ_{manifest["demonstration_seed"]}.npz','DRHQ',settings)
    rows=[]
    for name in ('Fixed','Actuated','QueueGreedy','DRHQ'):
        env=Intersection(settings)
        state=env.reset(arrivals=arrivals)
        done=False
        while not done:
            action=int(np.argmax(agent['actor'].forward(state)[0])) if name=='DRHQ' else baseline_action(name,env)
            state,_,done,metrics=env.step(action)
        rows.append({'controller':name,**metrics})
    print('Synthetic constant profile (N/S/E/W vehicles per second):',args.rates)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__=='__main__':main()
