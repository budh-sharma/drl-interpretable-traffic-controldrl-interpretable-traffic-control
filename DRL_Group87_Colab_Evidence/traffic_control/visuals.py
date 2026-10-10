"""Plots made from saved CSVs and model explanations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from .config import DIRECTIONS, FEATURE_NAMES

COLORS={'Fixed':'#94a3b8','Actuated':'#e79836','QueueGreedy':'#758344',
        'DQN':'#506cc8','DRQ':'#ae657d','DRSQ':'#55a0b4','DRHQ':'#008c78'}


def save_figure(fig,path):
    fig.savefig(path,dpi=140,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_figures(folder):
    folder=Path(folder)
    charts=folder/'figures'
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    summary=pd.read_csv(folder/'summary.csv')
    training=pd.read_csv(folder/'training.csv')
    for scenario in ('balanced','tidal'):
        group=summary[summary.scenario==scenario]
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(group.algorithm,group.delay_s,color=[COLORS[x] for x in group.algorithm])
        ax.set(ylabel='In-horizon queue delay / arrival (s)',title=f'{scenario.title()} traffic: frozen-policy evaluation')
        for i,value in enumerate(group.delay_s):ax.text(i,value+.4,f'{value:.1f}',ha='center')
        save_figure(fig,charts/f'delay_{scenario}.png')
        fig,ax=plt.subplots(figsize=(9,4))
        for controller in ('Fixed','Actuated','QueueGreedy','DRHQ'):
            trace=pd.read_csv(folder/f'trace_{scenario}_{controller}.csv')
            queue=trace[['q0','q1','q2','q3']].sum(axis=1)
            ax.plot(trace.second,queue.rolling(20,min_periods=1).mean(),label=controller,color=COLORS[controller])
        ax.set(xlabel='Simulation time (s)',ylabel='Total queued vehicles',title=f'{scenario.title()} example, traffic seed 90000')
        ax.legend(ncol=4)
        save_figure(fig,charts/f'queues_{scenario}.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for scenario,ax in zip(('balanced','tidal'),axes):
        for algorithm in ('DQN','DRQ','DRSQ','DRHQ'):
            group=training[(training.scenario==scenario)&(training.algorithm==algorithm)]
            avg=group.groupby('episode').delay_per_arrival_s.mean()
            ax.plot(avg.index,avg.values,label=algorithm,color=COLORS[algorithm])
        ax.set(title=f'{scenario.title()} training',xlabel='Episode',ylabel='Queue delay / arrival (s)')
        ax.legend()
    save_figure(fig,charts/'learning.png')
    fig,ax=plt.subplots(figsize=(9,4))
    for algorithm in ('DRQ','DRSQ','DRHQ'):
        avg=training[training.algorithm==algorithm].groupby('episode').replay_agreement.mean()
        ax.plot(avg.index,avg.values,label=algorithm,color=COLORS[algorithm])
    ax.set(xlabel='Episode',ylabel='Action agreement on replay minibatches',ylim=(0,1.02))
    ax.legend()
    save_figure(fig,charts/'agreement.png')
    examples=json.loads((folder/'examples.json').read_text())
    for i,example in enumerate(examples,1):
        terms=np.asarray(example['contributions']).sum(axis=1)
        fig,ax=plt.subplots(figsize=(9,4))
        left=np.zeros(2)
        for j,feature in enumerate(FEATURE_NAMES):
            ax.barh(['North / South','East / West'],terms[:,j],left=left,label=feature)
            left+=terms[:,j]
        ax.set(xlabel='Precedence score contribution',title=f'Example {i}: {example["scenario"]}, t={example["second"]} s')
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16))
        save_figure(fig,charts/f'explain_{i}.png')
    trace=pd.read_csv(folder/'trace_tidal_DRHQ.csv')
    fig,ax=plt.subplots(figsize=(10,3))
    phase=trace.phase.to_numpy()
    clearance=(trace.signal!='green').to_numpy()
    phase=phase.astype(float);phase[clearance]=np.nan
    ax.scatter(trace.second,phase,s=6,c=[COLORS['DRHQ'] if x==0 else COLORS['DQN'] for x in trace.phase])
    ax.set(yticks=[0,1],yticklabels=['NS green','EW green'],xlabel='Time (s)',title='Executed phase trace (clearance intervals appear as gaps)')
    save_figure(fig,charts/'phase_trace.png')
    fig,ax=plt.subplots(figsize=(8,4))
    chosen=summary[summary.algorithm=='DRHQ']
    ax.axvline(0,color='#777',lw=1)
    for y,(_,row) in enumerate(chosen.iterrows()):
        ax.plot([row.paired_ci_low_pct,row.paired_ci_high_pct],[y,y],lw=3,color=COLORS['DRHQ'])
        ax.scatter(row.improvement_pct,y,color=COLORS['DRHQ'])
    ax.set(yticks=[0,1],yticklabels=chosen.scenario.tolist(),xlabel='Delay reduction vs Actuated (%)',title='Paired demand-stream bootstrap, 95% intervals')
    save_figure(fig,charts/'paired_interval.png')


def make_animation(folder,scenario='tidal'):
    folder=Path(folder)
    data=pd.read_csv(folder/f'trace_{scenario}_DRHQ.csv').iloc[::20]
    fig,ax=plt.subplots(figsize=(6,3))
    bars=ax.bar(DIRECTIONS,[0,0,0,0],color=['#008c78']*4)
    ax.set_ylim(0,max(10,data[['q0','q1','q2','q3']].to_numpy().max()+3))
    ax.set_ylabel('Queued vehicles')
    title=ax.set_title('')
    def update(index):
        row=data.iloc[index]
        for i,bar in enumerate(bars):
            bar.set_height(row[f'q{i}'])
            active=(i<2)==(row.phase==0) and row.signal=='green'
            bar.set_color('#008c78' if active else '#c9635e')
        title.set_text(f'DRHQ: {int(row.second)} s, signal = {row.signal}')
        return list(bars)+[title]
    animation=FuncAnimation(fig,update,frames=len(data),interval=150)
    path=folder/'figures'/f'{scenario}_queues.gif'
    animation.save(path,writer=PillowWriter(fps=7))
    plt.close(fig)
    return path
