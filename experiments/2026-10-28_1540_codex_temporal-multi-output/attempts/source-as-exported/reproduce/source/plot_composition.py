"""Show original TRAIN support beside the prespecified optimizer exposure budget."""

from pathlib import Path
import hashlib
import json
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

F = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    composition = json.loads((F / 'results/cohort-composition.json').read_text())
    verification = json.loads((F / 'results/cohort-composition-verification.json').read_text())
    prepared = json.loads((F / 'results/prepared.json').read_text())
    assert verification['status'] == 'passed'
    assert sha(F / 'results/cohort-composition.json') == verification['result_sha256']
    endpoints = [
        'elution_observation',
        'ic50',
        'kd_via_ic50',
        'qualitative_binding',
        'kd_via_ec50',
        'kd',
        'half_life',
        'td50',
        'ec50',
        'koff',
        'kon',
    ]
    labels = [
        'MS observations',
        'IC50',
        'KD via IC50',
        'Qualitative binding',
        'KD via EC50',
        'KD',
        'Half-life',
        'Temperature (td50)',
        'EC50',
        'Off-rate',
        'On-rate',
    ]
    original = {
        (r['endpoint'], r['mhc_class']): r['n']
        for r in composition['populations']
        if r['split'] == 'train' and r['population'] == 'original'
    }
    exposure = {tuple(k.split(':')): v for k, v in prepared['epoch_quotas'].items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True, constrained_layout=True)
    for ax, data, title in zip(
        axes,
        (original, exposure),
        ('Original training observations', 'Planned optimizer exposures per seed'),
    ):
        for cls, shift, color in (('I', -0.18, '#277e8e'), ('II', 0.18, '#bc7028')):
            values = [data.get((e, cls), 0) for e in endpoints]
            positions = np.arange(len(endpoints)) + shift
            ax.barh(positions, values, height=0.34, color=color, label='MHC-' + cls)
            for y, value in zip(positions, values):
                ax.text(max(value * 1.08, 1.2), y, f'{value:,}', va='center', fontsize=8)
        ax.set_xscale('log')
        ax.set_xlim(1, 8_000_000)
        ax.set_title(title, loc='left', fontsize=12)
        ax.grid(axis='x', alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Count (log scale)')
    axes[0].set_yticks(np.arange(len(endpoints)), labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', frameon=False)
    fig.suptitle('Presto: measured support and the balanced training budget', fontsize=15)
    fig.supxlabel(
        'Original observations exclude generated comparators. Exposures include repeated labels and generated MS comparators.\n'
        'This is the frozen training design, not predictive performance.',
        fontsize=9,
    )
    for suffix in ('png', 'pdf'):
        fig.savefig(F / 'results' / ('training-composition.' + suffix), dpi=180)
    plt.close(fig)
    receipt = {
        'status': 'completed',
        'source_sha256': sha(Path(__file__)),
        'composition_sha256': verification['result_sha256'],
        'prepared_sha256': sha(F / 'results/prepared.json'),
        'created_unix': time.time(),
    }
    (F / 'results/composition-plot.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
