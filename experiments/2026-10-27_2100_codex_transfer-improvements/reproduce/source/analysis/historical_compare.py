"""Paired comparison with previously closed source-only and 1000-step controls."""

import json
import numpy as np
from compare import FAMILY, load, scores, table_csv, atomic, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
from verify import sha

PRIOR = FAMILY.parent / '2026-10-27_1930_codex_modal-cross-class'


def rows(family, variant, seed, split, kind):
    source = variant.rsplit('-', 1)[-1]
    own = source == 'joint' or source == kind
    filename = (
        f'{split}-predictions.jsonl.gz'
        if own
        else f'identical_weights_expanded_routing-{split}-predictions.jsonl.gz'
    )
    path = family / 'results' / f'{variant}-s{seed}' / filename
    data = [r for r in load(path) if r['qualifier'] == '=' and r['mhc_classes'] == [kind]]
    return sorted(data, key=lambda r: r['observation_id']), dict(path=str(path), sha256=sha(path))


def comparison(candidate, reference, split, kind):
    arrays = []
    provenance = []
    for family, variant in ((FAMILY, candidate), (PRIOR, reference)):
        for seed in (42, 43):
            records, receipt = rows(family, variant, seed, split, kind)
            provenance.append(receipt)
            if not arrays:
                ids = [r['observation_id'] for r in records]
                y = np.array([r['target'] for r in records])
                peptides = {}
                for i, row in enumerate(records):
                    peptides.setdefault(row['peptide'], []).append(i)
            assert [r['observation_id'] for r in records] == ids
            np.testing.assert_array_equal([r['target'] for r in records], y)
            arrays.append([r['prediction'] for r in records])
    p = np.array(arrays)
    rho, rmse = scores(y, p)
    point = np.array([rho.reshape(2, 2).mean(1), rmse.reshape(2, 2).mean(1)])
    groups = [np.array(v) for _, v in sorted(peptides.items())]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        ii = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        a, b = scores(y[ii], p[:, ii])
        a = a.reshape(2, 2).mean(1)
        b = b.reshape(2, 2).mean(1)
        draws.append([a[0] - a[1], b[0] - b[1]])
    draws = np.array(draws)
    out = []
    for i, metric in enumerate(('spearman', 'rmse')):
        lo, hi = np.nanquantile(draws[:, i], [0.025, 0.975])
        out.append(
            dict(
                candidate=candidate,
                reference=reference,
                split=split,
                evaluation_class=kind,
                metric=metric,
                candidate_mean=float(point[i, 0]),
                reference_mean=float(point[i, 1]),
                delta=float(point[i, 0] - point[i, 1]),
                ci95_low=float(lo),
                ci95_high=float(hi),
                peptide_clusters=len(groups),
                exact_rows=len(y),
            )
        )
    return out, provenance


def main():
    decision = json.loads((FAMILY / 'results/validation-decision.json').read_text())
    assert decision['test_used_for_decision'] is False
    all_rows = []
    inputs = []
    for split in ('validation', 'test'):
        for kind in ('I', 'II'):
            pairs = [
                ('pooled-natural-joint', 'shared-joint'),
                ('pooled-balanced-joint', f'shared-{kind}'),
                (f'interaction-{kind}', f'shared-{kind}'),
                (
                    f'interaction-{"II" if kind == "I" else "I"}',
                    f'shared-{"II" if kind == "I" else "I"}',
                ),
            ]
            for candidate, reference in pairs:
                result, provenance = comparison(candidate, reference, split, kind)
                all_rows.extend(result)
                inputs.extend(provenance)
    table_csv(FAMILY / 'results/historical-comparisons.csv', all_rows)
    atomic(
        FAMILY / 'results/historical-comparisons.json',
        dict(
            comparisons=all_rows,
            inputs=inputs,
            contract=(
                "paired peptide-cluster bootstrap, seed means not ensembles; "
                "previous inspected H100 development controls; no multiplicity correction"
            ),
        ),
    )
    prior = json.loads((PRIOR / 'results/seed-summary.json').read_text())
    table_csv(
        FAMILY / 'results/historical-context.csv',
        [r for r in prior if r['variant'] in ('shared-I', 'shared-II', 'shared-joint')],
    )
    print('Historical comparisons complete')


if __name__ == '__main__':
    main()
