"""Extract the fixed first validation checkpoint; never select from test results."""

from pathlib import Path
import csv
import hashlib
import json

F = Path(__file__).resolve().parents[1]


def main():
    rows, sources = [], {}
    for seed in (42, 43):
        path = F / f'results/provisional-validation-s{seed}-step1000.json'
        data = json.loads(path.read_text())
        assert data['evaluation_steps'] == data['history'][0]['steps'] == 1000
        assert data['status'] == 'provisional_worker_validation'
        assert data['test_metrics_available'] is False
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cls in ('I', 'II'):
            affinity = data['metrics']['validation']['ic50']['by_class'][cls]
            ligand = data['ligand_validation']['backgrounds']['same_protein']['metrics'][
                'elution_observation'
            ]['by_class'][cls]
            rows.append(
                dict(
                    seed=seed,
                    steps=1000,
                    split='validation',
                    mhc_class=cls,
                    ic50_n_exact=affinity['n'],
                    ic50_spearman=affinity['spearman'],
                    ic50_pearson=affinity['pearson'],
                    ic50_rmse_log10_nM=affinity['rmse'],
                    ligand_population='same_protein_unobserved_comparators',
                    ligand_n=ligand['n'],
                    ligand_auroc=ligand['auroc'],
                    ligand_auprc=ligand['auprc'],
                )
            )
    summary = dict(
        status='provisional_worker_validation',
        test_used=False,
        independent_per_example_verification=False,
        source_sha256=sources,
        rows=rows,
    )
    (F / 'results/provisional-first-validation.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    with (F / 'results/provisional-first-validation.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
