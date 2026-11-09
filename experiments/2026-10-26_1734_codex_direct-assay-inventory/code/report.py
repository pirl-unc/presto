"""Summarize the verified literal census without admitting or relabeling outcomes."""

from collections import defaultdict
import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]


def main():
    census = json.loads((FAMILY / 'results/census.json').read_text())
    replay = json.loads((FAMILY / 'results/independent-verification.json').read_text())
    assert replay['status'] == 'passed' and replay['source_rows_verified'] == 840805
    grouped = defaultdict(
        lambda: dict(raw_rows=0, positive_finite_rows=0, purified_metadata_candidates=0)
    )
    for row in census['joint_counts']:
        key = (row['response_measured'], row['measurement_units'])
        group = grouped[key]
        group['raw_rows'] += row['rows']
        positive = row['numeric_state'] == 'positive_finite'
        group['positive_finite_rows'] += row['rows'] * positive
        candidate = (
            positive
            and row['has_ptm'] is False
            and row['source'] in ('iedb', 'cedar')
            and row['mhc_class'] in ('I', 'II')
            and isinstance(row['assay_method'], str)
            and row['assay_method'].startswith('purified MHC/')
        )
        group['purified_metadata_candidates'] += row['rows'] * candidate
    labels = [
        dict(response_measured=key[0], measurement_units=key[1], **value)
        for key, value in sorted(grouped.items(), key=lambda item: -item[1]['raw_rows'])
    ]
    (FAMILY / 'results/label-summary.json').write_text(json.dumps(labels, indent=2) + '\n')
    launch = json.loads((FAMILY / 'reproduce/launch.json').read_text())
    receipts = {}
    for name in ('inventory', 'independent-replay'):
        paths = sorted((FAMILY / 'reproduce').glob(name + '-*.json'))
        receipts[name] = json.loads(paths[-1].read_text())
    lines = [
        '# Literal direct-assay source inventory',
        '',
        '**Closed: both independent scans agree on all840,805 source rows and384 joint cells.**',
        'Registered and executed2026-10-26 by Codex / GPT-6. Zero rows admitted for fitting.',
        '',
        'Frozen source: Hitlist ci-corpus-v4, the existing2026-10-16 export; this is not',
        'a latest-release claim. The original binding.parquet (814,268 rows) and',
        'other_assays.parquet (26,537 rows) are hash-verified against the pinned',
        '[source contract](reproduce/plan.md), under the existing local release path',
        '`/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z/`.',
        '',
        'The census preserves literal response, method, unit, qualifier, source, class,',
        'PTM and binding/MS flags, including null separately from empty strings.',
        'Numeric state is only missing/nonfinite/nonpositive/positive finite. No',
        'qualifier interpretation, unit conversion, target mapping, sequence/groove',
        'resolution, deduplication, split assignment or biological admission occurred.',
        '',
        (
            '| Literal source response | Unit | Raw rows | Positive finite | '
            'Purified metadata candidates |'
        ),
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in labels:
        lines.append(
            f"| {row['response_measured']} | {row['measurement_units'] or '(empty)'} | "
            f"{row['raw_rows']:,} | {row['positive_finite_rows']:,} | "
            f"{row['purified_metadata_candidates']:,} |"
        )
    lines += [
        '',
        'The last column is a metadata-only screen: positive finite value,',
        'has_ptm=False, IEDB/CEDAR, classI/II and a literal method beginning',
        '`purified MHC/`. It does **not** check sequence chemistry, human restriction,',
        'explicit class-II partners, catalog agreement, raw numeric/bound text, original',
        'assay lineage, duplicates or leakage. Counts cannot be called training support.',
        '',
        'Decision: qualify the literal **dissociation constant KD** source next, then',
        'assess incremental retained-IC50 performance only after the fixed affinity',
        'comparison is reviewed. Preserve `KD (~IC50)` and `KD (~EC50)` as separate',
        'proxy families. Likewise, `50% dissociation temperature` requires assay-level',
        'qualification before any melting-temperature mapping; neither qualitative',
        'binding values nor off/on rates are silently reassigned to another endpoint.',
        '',
        'Verification uses a second scan with Arrow vector predicates/group-by and2048',
        'row batches, independent of the first Python row-wise4096-batch aggregation.',
        'All joint counts, null/empty counts, distinct nonempty assay/reference ID',
        'cardinalities, parquet totals and source hashes agree. The full source schema',
        'and per-source summaries are in [census.json](results/census.json); every cell',
        'is in [joint-counts.csv](results/joint-counts.csv), with JSON-encoded field',
        'values distinguish null/empty/false. [Replay](results/independent-verification.json).',
        '',
        'Execution commit: `' + launch['git_commit'] + '`; exact dirty state, invocation,',
        'environment and package versions are recorded in [launch.json](reproduce/launch.json).',
        'Only independent experiment artifacts were untracked; existing model/runtime',
        'code stayed unchanged. Reproduction uses [launch.sh](reproduce/launch.sh)',
        'at the executed code commit in a fresh checkout with the same pinned source',
        'files. Existing results are protected from overwrite. Code snapshots and the',
        'saved-output report source are under `reproduce/source/`.',
        '',
        'Local CPU, one Arrow/BLAS thread; existing Python3.12.6, no installs, network,',
        'cloud or GPU. No model, pretraining, optimizer, loss weights, synthetic data,',
        'assay-output mapping or selected condition exists. Predictive metrics and',
        'validation/test dumps are inapplicable because no cohort/splits or model were',
        'created. This source inventory does not qualify any scientific endpoint.',
        '',
        '| Operation | Elapsed seconds | Peak RSS bytes | Status |',
        '| --- | ---: | ---: | --- |',
    ]
    for name, r in receipts.items():
        lines.append(
            f"| {name} | {r['elapsed_seconds']:.6f} | "
            f"{r['process_peak_rss_bytes']:,} | {r['status']} |"
        )
    lines += [
        '',
        'Both scans stayed within the registered300s/3GiB process ceilings and',
        'available-RAM/disk floors. The raw sources remain at the pinned original',
        'paths; no new large cohort was materialized alongside IC50 training.',
        '',
        'Status: closed source inventory. Next step: strict direct-KD admission.',
        'Open questions: original-assay interpretation, deduplication, explicit partner',
        'coverage, leakage control and useful retained-task fitting. PR131 remains draft.',
        '',
    ]
    text = '\n'.join(lines)
    (FAMILY / 'README.md').write_text(text)
    log = ROOT / 'experiments/experiment_log.md'
    marker = '## Direct-assay source inventory — 2026-10-26_1734_codex'
    if marker in log.read_text():
        raise FileExistsError('canonical entry already exists; review before replacing')
    with log.open('a') as stream:
        stream.write(
            '\n\n'
            + marker
            + f'\n\nExperiment directory: [{FAMILY.name}]({FAMILY.name}/README.md).\n\n'
            + text.split('\n', 1)[1]
            .replace('(results/', f'({FAMILY.name}/results/')
            .replace('(reproduce/', f'({FAMILY.name}/reproduce/')
        )
    print('Closed verified literal census; next admission candidate is direct KD', flush=True)


if __name__ == '__main__':
    main()
