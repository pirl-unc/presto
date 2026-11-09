"""Metadata-only holdout projection of the pinned parent cohort."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def records(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def prepare(parent, old_root, destination, policy):
    from presto.data.assay_measurement import normalize_assay_method

    old_root, destination = Path(old_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    binding = {
        'parent_manifest_sha256': hashlib.sha256(
            json.dumps(parent, sort_keys=True).encode()
        ).hexdigest(),
        'policy_sha256': hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
    }
    if (destination / 'prepared.json').exists():
        result = read(destination / 'prepared.json')
        assert result['binding'] == binding, 'preparation contract changed'
        for source in result['sources']:
            assert sha(source['path']) == source['sha256']
        return result
    sources = [(source, old_root / Path(source['path']).name) for source in parent['sources']]
    old_heldout, extra = set(), {'validation': set(), 'test': set()}
    counts, selected_counts = Counter(), Counter()
    for source, path in sources:
        assert sha(path) == source['sha256'], path
        for row in records(path):
            counts[row['split']] += 1
            if row['split'] != 'train':
                old_heldout.add(row['peptide'])
            elif row['group_id'] in policy['groups']:
                split = policy['groups'][row['group_id']]['split']
                extra[split].add(row['peptide'])
                selected_counts[row['group_id']] += 1
    assert set(selected_counts) == set(policy['groups']), 'selected components absent'
    heldout = old_heldout | extra['validation'] | extra['test']
    projected, disposition, after, methods = [], Counter(), Counter(), Counter()
    split_peptides, split_groups = {s: set() for s in counts}, {s: set() for s in counts}
    for source, path in sources:
        target = destination / path.name
        n = 0
        with target.open('w') as out:
            for row in records(path):
                if row['split'] == 'train':
                    group = policy['groups'].get(row['group_id'])
                    if group:
                        split = group['split']
                        if row['peptide'] in old_heldout or (
                            split == 'validation' and row['peptide'] in extra['test']
                        ):
                            disposition['extra_overlap_removed'] += 1
                            continue
                        row['split'] = split
                        row['provenance']['assay_holdout_kind'] = 'additional_fluorescence_study'
                        disposition['moved_' + split] += 1
                    elif row['peptide'] in heldout:
                        disposition['train_peptide_embargo'] += 1
                        continue
                    else:
                        disposition['retained_train'] += 1
                else:
                    disposition['original_heldout_retained'] += 1
                method = normalize_assay_method(row['provenance'].get('raw_record', {}))
                row['assay_method'] = asdict(method)
                cls = ';'.join(sorted({m['mhc_class'] for m in row['mhc']}))
                after[f"{row['split']}:{row['endpoint']}:{cls}"] += 1
                methods[
                    f"{row['split']}:{row['endpoint']}:{cls}:{method.preparation}/{method.mechanism}/{method.detector}:{method.review_status}"
                ] += 1
                split_peptides[row['split']].add(row['peptide'])
                split_groups[row['split']].add(row['group_id'])
                out.write(json.dumps(row, sort_keys=True) + '\n')
                n += 1
        if n:
            projected.append(
                {
                    **source,
                    'path': str(target),
                    'sha256': sha(target),
                    'split_policy': source['split_policy']
                    + ('; additional fluorescence study holdouts and cross-output peptide embargo'),
                }
            )
    assert sum(disposition.values()) == sum(counts.values())
    assert disposition['original_heldout_retained'] == counts['validation'] + counts['test']
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        assert not split_peptides[a] & split_peptides[b], (a, b, 'peptide overlap')
        assert not split_groups[a] & split_groups[b], (a, b, 'study overlap')
    manifests = {}
    for mode in ('pooled', 'calibration', 'factorized'):
        for seed in (42, 43):
            condition = f'{mode}-s{seed}'
            manifest = {
                **parent,
                'model': {
                    **parent['model'],
                    'assay_mode': mode,
                    'assay_tracks': [
                        dict(
                            endpoint='ic50',
                            preparation='purified',
                            mechanism='competitive',
                            detector=d,
                        )
                        for d in ('radioactivity', 'fluorescence')
                    ],
                },
                'sources': projected,
                'training': {
                    **parent['training'],
                    'seed': seed,
                    'max_steps': 4000,
                    'train_evaluation_max_rows': 2048,
                },
            }
            file = destination / (condition + '.json')
            write(file, manifest)
            manifests[condition] = {'path': str(file), 'sha256': sha(file)}
    result = dict(
        binding=binding,
        status='prepared',
        sources=projected,
        manifests=manifests,
        input_counts=dict(counts),
        disposition=dict(disposition),
        counts=dict(after),
        methods=dict(methods),
        selected_component_rows=dict(selected_counts),
        overlap_audit='passed: peptide and study disjoint across all three splits',
        holdout_policy=policy,
        training=manifest['training'],
    )
    write(destination / 'prepared.json', result)
    return result
