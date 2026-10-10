"""Check that the saved report agrees with its raw data and checkpoints."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from traffic_control.config import Settings
from traffic_control.evaluation import paired_summary
from traffic_control.training import load_agent


def validate(folder='results'):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    evaluation=pd.read_csv(folder/'evaluation.csv')
    summary=pd.read_csv(folder/'summary.csv')
    recomputed=paired_summary(evaluation)
    np.testing.assert_allclose(summary.select_dtypes('number'),recomputed.select_dtypes('number'))
    assert np.all(evaluation.arrived == evaluation.departed+evaluation.pending)
    assert not set(manifest['evaluation_seeds']) & {
        10000+seed*1000+episode for seed in manifest['training_seeds']
        for episode in range(manifest['settings']['episodes'])}
    for (scenario,seed),group in evaluation.groupby(['scenario','demand_seed']):
        assert group.arrived.nunique()==1, 'Controllers received different arrival totals'
    cfg=Settings(**manifest['settings'])
    for file in (folder/'models').glob('*.npz'):
        algorithm=file.stem.split('_')[0]
        agent=load_agent(file,algorithm,cfg)
        assert all(np.all(np.isfinite(v)) for v in agent['teacher'].parameters.values())
        if agent['actor']:
            assert all(np.all(np.isfinite(v)) for v in agent['actor'].parameters.values())
    for example in json.loads((folder/'examples.json').read_text()):
        np.testing.assert_allclose(np.asarray(example['contributions']).sum(axis=(1,2)),example['scores'])
        assert example['action']==int(np.argmax(example['scores']))
    for name,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
    print(f'Validated {len(evaluation)} frozen rollouts, summaries, example scores and model checkpoints.')
    return True

if __name__=='__main__': validate()
