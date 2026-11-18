"""Common-cohort model comparisons; selected and terminal are not replicates."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

E = Path(__file__).resolve().parents[1]
ARMS = ('continued', 'paired', 'paired-junction')
LABELS = ('Continued', '+ Matched\nsupply loss', '+ Boundary\nmodule')
COLORS = {42: '#2166ac', 43: '#b76516'}


def figure(data, populations, metric, readout, title, filename):
    fig, axes = plt.subplots(len(populations), 2, figsize=(11, 3.1 * len(populations)), squeeze=False)
    for row, (population, label) in enumerate(populations):
        for column, cls in enumerate(('I', 'II')):
            ax = axes[row, column]
            cell = data[(data.population == population) & (data['class'] == cls) & (data.readout == readout)]
            baseline = cell[cell.model == 'parent'][metric].item()
            ax.axhline(baseline, color='#777777', linestyle='--', linewidth=1.3, label='Parent')
            values = [baseline]
            for seed in (42, 43):
                selected, terminal = [], []
                for arm in ARMS:
                    name = f'{arm}-s{seed}-n2000'
                    selected.append(cell[cell.model == name][metric].item())
                    terminal.append(cell[cell.model == name + '-terminal'][metric].item())
                ax.plot(range(3), selected, color=COLORS[seed], marker='o', linewidth=1.3, markersize=6, label=f'Seed {seed}: selected')
                ax.scatter([x + .06 for x in range(3)], terminal, color=COLORS[seed], marker='x', s=42, linewidths=1.7, label=f'Seed {seed}: terminal')
                values += selected + terminal
            pad = max(.015, (max(values) - min(values)) * .2)
            ax.set_ylim(max(0, min(values) - pad), min(1, max(values) + pad))
            ax.set_title(f'{label} · Class {cls}', fontsize=11, loc='left', fontweight='bold')
            ax.set_xticks(range(3), LABELS, fontsize=9)
            ax.set_ylabel('AUROC' if metric == 'auroc' else 'Within-pair win rate', fontsize=10)
            ax.grid(axis='y', alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='y', labelsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16, x=.06, ha='left', fontweight='bold')
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, .025), frameon=False, fontsize=9)
    fig.text(.06, .012, 'Identical validation rows within each panel; different cohorts across rows. Terminal = fixed update 2,000. TEST sealed.', fontsize=8, color='#444444')
    fig.tight_layout(rect=(0, .10, 1, .96), h_pad=2)
    out = E / 'results/figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / (filename + '.png'), dpi=180)
    fig.savefig(out / (filename + '.pdf'))
    plt.close(fig)


def main():
    data = pd.read_csv(E / 'results/analysis/metrics.csv')
    cohorts = [('expanded', 'Broad ordinary validation'), ('matched', 'Historical nM-matched validation'), ('rank-matched', 'Hybrid BA-rank validation')]
    figure(data, cohorts, 'auroc', 'prediction', 'Presentation performance on fixed cohorts', 'presentation-auroc')
    figure(data, cohorts[1:], 'pair_win', 'prediction', 'Ordering observed peptides above their matched backgrounds', 'matched-pair-wins')
    figure(data, cohorts[1:], 'pair_win', 'processing_supply', 'Processing-supply readout on matched pairs', 'supply-pair-wins')


if __name__ == '__main__':
    main()
