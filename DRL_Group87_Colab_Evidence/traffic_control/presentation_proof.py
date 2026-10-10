"""Small evidence figures for the portrait screenshot spaces in the PPT."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GREEN = '#107c41'
INK = '#192a30'
GRAY = '#647477'
ORANGE = '#d98c35'


def canvas(title, source, folder='results'):
    folder = Path(folder)
    fig = plt.figure(figsize=(4.4, 5.5), facecolor='white')
    fig.text(.07, .94, title, fontsize=16, weight='bold', color=GREEN)
    fig.text(.07, .89, 'Group 87 · measured notebook output', fontsize=9, color=GRAY)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    file = folder / source
    digest = hashlib.sha256(file.read_bytes()).hexdigest()[:10]
    fig.text(.07, .065, f'{source} · SHA256 {digest}', fontsize=8, color=GRAY)
    fig.text(.07, .035, stamp, fontsize=8, color=GRAY)
    return fig


def line(fig, y, text, size=12, color=INK, bold=False):
    fig.text(.07, y, text, fontsize=size, color=color,
             weight='bold' if bold else 'normal')


def save(fig, name, folder='results'):
    target = Path(folder) / 'ppt_proof'
    target.mkdir(exist_ok=True)
    # Fixed portrait canvas preserves the PPT slot's proportions.
    path = target / (name + '.png')
    fig.savefig(path, dpi=200, facecolor='white')
    fig.savefig(target / (name + '.svg'), facecolor='white')
    plt.close(fig)
    return path


def tests_panel(folder='results'):
    text = (Path(folder) / 'tests.txt').read_text()
    match = re.search(r'Ran (\d+) tests?', text)
    if not match or '\nOK' not in text:
        raise RuntimeError('No successful test run found. Do not label it passed.')
    count = int(match.group(1))
    fig = canvas('Correctness checks', 'tests.txt', folder)
    line(fig, .73, f'{count} tests passed', 26, GREEN, True)
    line(fig, .62, 'Clearance and vehicle accounting')
    line(fig, .54, 'Q and all actor gradients')
    line(fig, .46, 'Terminal targets and checkpoints')
    line(fig, .38, 'Frozen runs and unfinished queues')
    line(fig, .23, 'Executed unittest suite', 12, GREEN, True)
    line(fig, .17, 'Full test log remains in tests.txt.', 10)
    return save(fig, 's08_tests', folder)


def training_panel(folder='results'):
    folder = Path(folder)
    data = pd.read_csv(folder / 'training.csv')
    seeds = sorted(data.train_seed.unique())
    fig = canvas('Training completed', 'training.csv', folder)
    line(fig, .79, f'{len(data)} episodes · {len(seeds)} training seeds', 14, GREEN, True)
    ax = fig.add_axes([.16, .34, .78, .37])
    colors = {'DQN':'#506cc8', 'DRQ':'#ae657d', 'DRSQ':'#55a0b4', 'DRHQ':GREEN}
    for algorithm, group in data.groupby('algorithm', sort=False):
        means = group.groupby('episode').delay_per_arrival_s.mean()
        ax.plot(means.index, means.values, label=algorithm, color=colors[algorithm], lw=1.8)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Delay / arrival (s)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, ncol=2, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    line(fig, .22, 'Mean across seeds and training demands', 10)
    line(fig, .16, 'Evaluation follows with fixed parameters.', 10)
    return save(fig, 's09_training', folder)


def measured_rows(scenario, folder='results'):
    folder = Path(folder)
    summary = pd.read_csv(folder / 'summary.csv')
    rows = summary[summary.scenario == scenario].set_index('algorithm')
    evaluation = pd.read_csv(folder / 'evaluation.csv')
    outcomes = evaluation[evaluation.scenario == scenario].groupby('algorithm')[
        ['pending', 'throughput_fraction']].mean()
    a, g = rows.loc['Actuated'], rows.loc['DRHQ']
    reduction = 100 * (a.delay_s - g.delay_s) / a.delay_s
    np.testing.assert_allclose(reduction, g.improvement_pct)
    return a, g, outcomes, reduction


def balanced_panel(folder='results'):
    a, g, outcomes, reduction = measured_rows('balanced', folder)
    fig = canvas('Balanced demand result', 'summary.csv', folder)
    ax = fig.add_axes([.17, .56, .76, .24])
    bars = ax.bar(['Actuated', 'DRHQ'], [a.delay_s, g.delay_s], color=[ORANGE, GREEN])
    ax.set_ylim(0, max(a.delay_s, g.delay_s) * 1.3)
    ax.set_ylabel('Delay / arrival (s)', fontsize=10)
    ax.tick_params(labelsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, value in zip(bars, [a.delay_s, g.delay_s]):
        ax.text(bar.get_x()+bar.get_width()/2, value+.25, f'{value:.3f}', ha='center', fontsize=12)
    line(fig, .46, f'{reduction:.2f}% lower delay', 19, GREEN, True)
    line(fig, .40, f'({a.delay_s:.3f} − {g.delay_s:.3f}) / {a.delay_s:.3f} × 100', 11)
    line(fig, .31, '                    Actuated     DRHQ', 11, bold=True)
    line(fig, .255, f'Pending       {outcomes.loc["Actuated", "pending"]:.2f}          {outcomes.loc["DRHQ", "pending"]:.2f}', 11)
    line(fig, .20, f'Throughput  {100*outcomes.loc["Actuated", "throughput_fraction"]:.2f}%      {100*outcomes.loc["DRHQ", "throughput_fraction"]:.2f}%', 11)
    line(fig, .125, 'Pending and throughput: evaluation.csv means.', 9)
    return save(fig, 's10_balanced', folder)


def decision_panel(index, folder='results'):
    folder = Path(folder)
    example = json.loads((folder / 'examples.json').read_text())[index]
    terms = np.asarray(example['contributions']).sum(axis=1)
    scores = np.asarray(example['scores'])
    np.testing.assert_allclose(terms.sum(axis=1), scores)
    assert example['action'] == int(np.argmax(scores))
    names = ['Queued', 'Approach', 'Total wait', 'Mean wait', 'Queue/lane', 'Speed']
    colors = ['#107c41','#e2a14b','#4986aa','#7e75a4','#65b78b','#c4b170']
    fig = canvas('One executed decision', 'examples.json', folder)
    line(fig, .79, f'{example["scenario"].title()} · t = {example["second"]} s', 14, GREEN, True)
    ax = fig.add_axes([.18, .52, .75, .18])
    left = np.zeros(2)
    for j in range(6):
        ax.barh(['NS', 'EW'], terms[:, j], left=left, color=colors[j], label=names[j])
        left += terms[:, j]
    ax.invert_yaxis()
    ax.set_xlabel('Feature contributions to action score', fontsize=9)
    ax.tick_params(labelsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.45, -.38), ncol=3, fontsize=8, frameon=False)
    line(fig, .31, f'NS score: {scores[0]:.4f}', 15, bold=True)
    line(fig, .25, f'EW score: {scores[1]:.4f}', 15, bold=True)
    choice = 'North / South' if example['action'] == 0 else 'East / West'
    line(fig, .17, f'Winner: {choice}', 17, GREEN, True)
    line(fig, .115, 'Contribution sums and argmax verified.', 10)
    return save(fig, 's11_decision' if index == 0 else 's12_decision', folder)


def reversal_panel(folder='results'):
    folder = Path(folder)
    a, g, outcomes, reduction = measured_rows('tidal', folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    midpoint = manifest['settings']['horizon'] / 2
    fig = canvas('Demand reversal result', 'summary.csv', folder)
    ax = fig.add_axes([.16, .55, .79, .25])
    for name, color in [('Actuated', ORANGE), ('DRHQ', GREEN)]:
        trace = pd.read_csv(folder / f'trace_tidal_{name}.csv')
        queue = trace[['q0', 'q1', 'q2', 'q3']].sum(axis=1).rolling(20, min_periods=1).mean()
        ax.plot(trace.second, queue, label=name, color=color, lw=1.5)
    ax.axvline(midpoint, color=GRAY, ls='--', lw=1)
    ax.set_xlabel('Time (s), reversal at '+str(int(midpoint)), fontsize=10)
    ax.set_ylabel('Queued vehicles', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    line(fig, .44, f'{reduction:.2f}% lower mean delay', 18, GREEN, True)
    line(fig, .365, f'Delay: {a.delay_s:.3f} → {g.delay_s:.3f} s/arrival', 12)
    line(fig, .29, f'Pending: {outcomes.loc["Actuated", "pending"]:.2f} → {outcomes.loc["DRHQ", "pending"]:.2f}', 12)
    line(fig, .215, f'Throughput: {100*outcomes.loc["Actuated", "throughput_fraction"]:.2f}% → {100*outcomes.loc["DRHQ", "throughput_fraction"]:.2f}%', 12)
    line(fig, .13, 'Trace: one demand seed. Metrics: all held-out runs.', 9)
    return save(fig, 's12_reversal', folder)


def checks_panel(folder='results'):
    folder = Path(folder)
    proof = json.loads((folder / 'proof_summary.json').read_text())
    checks = json.loads((folder / 'monotonicity.json').read_text())
    count = sum(v['checked_derivatives'] for v in checks.values())
    violations = sum(v['sign_violations'] for v in checks.values())
    if not all([proof['tests_reported_ok'], proof['conservation_all_rollouts'],
                proof['diagnostic_parameters_unchanged']]) or violations:
        raise RuntimeError('An evidence check failed. Inspect the detailed output.')
    fig = canvas('Verified experiment', 'proof_summary.json', folder)
    test_count = re.search(r'Ran (\d+) tests?', (folder / 'tests.txt').read_text()).group(1)
    line(fig, .83, f'{test_count} tests passed', 12, GREEN, True)
    line(fig, .76, f'{proof["training_episodes"]} training episodes', 18, GREEN, True)
    line(fig, .66, f'{proof["frozen_evaluation_rollouts"]} frozen rollouts', 18, GREEN, True)
    line(fig, .56, f'{proof["diagnostic_rollouts"]} diagnostic rollouts', 18, GREEN, True)
    line(fig, .43, 'Vehicle conservation: passed', 12)
    line(fig, .355, 'Diagnostic parameters unchanged', 12)
    line(fig, .255, f'{count:,} derivative checks', 16, GREEN, True)
    line(fig, .18, f'{violations} sign violations at saved checkpoints', 11)
    line(fig, .115, 'Derivative scope: fixed clearance category.', 9)
    return save(fig, 's13_checks', folder)


def result_method_panel(folder='results'):
    a, g, _, reduction = measured_rows('balanced', folder)
    fig = canvas('How we obtained the result', 'summary.csv', folder)
    line(fig, .77, 'Same seeded arrivals', 17, GREEN, True)
    line(fig, .705, 'Frozen Actuated and DRHQ rollouts', 12)
    line(fig, .60, 'Delay = queue waiting / all arrivals', 12, bold=True)
    line(fig, .54, 'Waiting accumulated inside the horizon.', 10)
    line(fig, .435, 'Average training seeds per demand,', 11)
    line(fig, .38, 'then average held-out demands.', 11)
    line(fig, .27, f'({a.delay_s:.3f} − {g.delay_s:.3f})', 16, GREEN, True)
    line(fig, .205, f'/ {a.delay_s:.3f} × 100 = {reduction:.2f}%', 16, GREEN, True)
    line(fig, .12, 'CI: paired demand variation, trained models fixed.', 9)
    return save(fig, 's13_result_method', folder)
