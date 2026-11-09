"""Exact source-review delta; full common preflight still runs in each fit worker."""

from collections import Counter
from pathlib import Path
import copy
import gzip
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'
REVISION = 'kinetic-review-v2'
app = modal.App('presto-temporal-reviewed-cohort')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12').uv_pip_install(
    'torch==2.14.1',
    'numpy==2.5.4',
    'mhcgnomes==3.64.6',
    'mhcseqs==2.6.13',
    'pyyaml==6.0.3',
    'tqdm==4.70.1',
    'psutil==7.2.2',
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    U = F.parents[1] / 'artifacts' / FID / 'upload'
    image = (
        image.add_local_dir(U / 'presto', remote_path='/runtime/presto')
        .add_local_file(U / 'runtime-pins.json', remote_path='/bundle/runtime-pins.json')
        .add_local_file(F / 'code/prepare_worker.py', remote_path='/bundle/prepare_worker.py')
        .add_local_file(F / 'code/partition.py', remote_path='/bundle/partition.py')
        .add_local_file(
            F / 'analysis/cohort_composition.py', remote_path='/bundle/cohort_composition.py'
        )
    )


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_delta(old, new, removed_ids):
    """Compare retained bytes/order independently from the admission-review helper."""
    removed = []
    with Path(old).open('rb') as before, Path(new).open('rb') as after:
        for line in before:
            row = json.loads(line)
            if row['observation_id'] in removed_ids:
                removed.append(row['observation_id'])
            else:
                assert after.readline() == line, 'unreviewed row bytes or order changed'
        assert after.read() == b'', 'unexpected additional rows'
    assert set(removed) == removed_ids and len(removed) == len(removed_ids)


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    timeout=14400,
    nonpreemptible=True,
    max_containers=1,
    volumes={'/volume': volume},
)
def revise():
    sys.path[:0] = ['/runtime', '/bundle']
    from presto.data.assay_reviews import assay_review_reason
    from presto.data.modular import Observation, fit_target_scales, observation_target
    from presto.training.modular_sampling import weighted_quotas
    from prepare_worker import weights

    started = time.time()
    base = Path('/volume') / FID
    durable = base / 'revisions' / REVISION
    pins = read('/bundle/runtime-pins.json')
    for name, digest in pins['files'].items():
        assert sha(Path('/runtime/presto') / name) == digest
    policy_path = Path('/runtime/presto/data/assay_measurement_reviews.json')
    policy_sha = sha(policy_path)
    if (durable / 'preparation-result.json').exists():
        result = read(durable / 'preparation-result.json')
        assert result['source_review']['policy_sha256'] == policy_sha
        return result
    prior = read(base / 'preparation-result.json')
    assert prior['verification']['status'] == 'passed'
    original = base / 'prepared'
    prepared = read(original / 'prepared.json')
    output = Path('/runs') / FID / 'prepared'
    output.mkdir(parents=True, exist_ok=False)
    original_manifest = read(original / 'wide-multi-s42.json')
    decisions = {row['assay_iri']: row for row in read(policy_path)['entries']}
    assert len(decisions) == 20
    sources, removed, input_receipts = [], [], {}
    for spec in original_manifest['sources']:
        path = Path(spec['path'])
        assert path.parent == output
        old = original / path.name
        assert sha(old) == spec['sha256']
        if path.name == 'hitlist-v4-iedb-train.jsonl':
            with old.open('rb') as source, path.open('xb') as target:
                for line in source:
                    row = json.loads(line)
                    iri = row['provenance']['raw_record']['assay_iri']
                    reason = assay_review_reason(iri)
                    if reason:
                        expected = decisions[iri]['observed_export']
                        assert row['split'] == 'train'
                        assert (row['value'], row['unit'], row['qualifier']) == (
                            expected['value'],
                            expected['unit'],
                            expected['qualifier'],
                        )
                        removed.append(row)
                    else:
                        target.write(line)
            ids = {row['observation_id'] for row in removed}
            assert len(removed) == len(ids) == 20
            assert {row['provenance']['raw_record']['assay_iri'] for row in removed} == set(
                decisions
            )
            verify_delta(old, path, ids)
        else:
            shutil.copyfile(old, path)
            assert sha(path) == spec['sha256']
        new = {**spec, 'sha256': sha(path)}
        sources.append(new)
        input_receipts[path.name] = dict(before_sha256=spec['sha256'], after_sha256=new['sha256'])
        print('Verified source delta', path.name, flush=True)
    ids = {row['observation_id'] for row in removed}
    p = copy.deepcopy(prepared)
    removed_counts = Counter()
    for row in removed:
        cls, endpoint = row['mhc'][0]['mhc_class'], row['endpoint']
        for field, key in (
            ('counts', f'train:{endpoint}:{cls}:{row["qualifier"]}'),
            ('train_strata', f'{endpoint}:{cls}'),
            ('physical_observations', f'train:{endpoint}'),
        ):
            p[field][key] -= 1
            if p[field][key] == 0:
                del p[field][key]
        reason = assay_review_reason(row['provenance']['raw_record']['assay_iri'])
        removed_counts[reason] += 1
    for reason, n in removed_counts.items():
        p['dispositions']['train:source_review:' + reason] = n
    strata = {tuple(k.split(':')): n for k, n in p['train_strata'].items()}
    sampling = weights(strata)
    quotas = weighted_quotas(strata, sampling)
    p['epoch_quotas'] = {':'.join(k): n for k, n in sorted(quotas.items())}
    p['weighted_epoch_examples'] = sum(quotas.values())
    p['steps'] = max(6000, math.ceil(p['weighted_epoch_examples'] / 128))
    endpoints = sorted({key[0] for key in strata})
    assert set(prepared['model']['endpoints']) - set(endpoints) == {'kon'}
    p['model']['endpoints'] = endpoints
    exact_values = {}

    def physical_rows():
        with (output / 'hitlist-v4-iedb-train.jsonl').open() as stream:
            for line in stream:
                row = Observation.from_dict(json.loads(line))
                if row.qualifier == '=' and row.endpoint != 'qualitative_binding':
                    exact_values.setdefault(row.endpoint, []).append(observation_target(row))
                yield row

    scales = fit_target_scales(physical_rows())
    assert all(len(set(values)) >= 2 for values in exact_values.values())
    for endpoint, scale in scales.items():
        values = exact_values[endpoint]
        assert scale.exact_count == len(values)
        assert math.isclose(scale.mean, statistics.fmean(values), abs_tol=1e-10)
        assert math.isclose(scale.scale, max(scale.floor, statistics.pstdev(values)), abs_tol=1e-10)
    assert set(scales) == set(endpoints) - {'elution_observation', 'qualitative_binding'}
    p['expected_scales'] = {key: vars(value) for key, value in scales.items()}
    p['manifests'] = {}
    for seed in (42, 43):
        name = 'wide-multi-s' + str(seed)
        manifest = read(original / (name + '.json'))
        manifest.update(model=p['model'], sources=sources)
        manifest['training'].update(
            max_steps=p['steps'],
            max_epochs=1,
            stratum_weights=[dict(values=list(k), weight=v) for k, v in sorted(sampling.items())],
        )
        path = output / (name + '.json')
        write(path, manifest)
        p['manifests'][name] = dict(
            path=str(path), sha256=sha(path), seed=seed, max_steps=p['steps']
        )
    # Study selection and every negative draw remain exactly the original frozen ones.
    shutil.copyfile(original / 'study-selection.json', output / 'study-selection.json')
    membership_removed = set()
    with (
        gzip.open(original / 'membership.jsonl.gz', 'rb') as before,
        gzip.open(output / 'membership.jsonl.gz', 'wb', compresslevel=1) as after,
    ):
        for line in before:
            identifier = json.loads(line)['id']
            if identifier in ids:
                membership_removed.add(identifier)
            else:
                after.write(line)
    assert membership_removed == ids
    write(
        output / 'source-review-dispositions.json',
        [
            dict(
                observation_id=row['observation_id'],
                assay_iri=row['provenance']['raw_record']['assay_iri'],
                reason=assay_review_reason(row['provenance']['raw_record']['assay_iri']),
                original=row,
            )
            for row in removed
        ],
    )
    train_counts = Counter()
    for (endpoint, _cls), count in strata.items():
        train_counts[endpoint] += count
    write(
        output / 'endpoint-support.json',
        dict(
            train_counts=dict(train_counts),
            allocated=endpoints,
            unsupported=['kon'],
            rule='reviewed TRAIN support; quantitative heads need distinct exact values',
        ),
    )
    write(
        output / 'preflight.json',
        dict(
            status='pending_in_trainers',
            expected_scales=p['expected_scales'],
            original_full_preflight_sha256=sha(original / 'preflight.json'),
            explanation='CPU verified the 20-row delta and recomputed TRAIN scales/quotas. '
            'Each fit still executes full canonical preflight before any optimizer update.',
        ),
    )
    verification = {
        **prior['verification'],
        'original_measurements': prior['verification']['original_measurements'] - 20,
        'review_revision': REVISION,
        'exact_reviewed_row_removal': True,
        'removed_rows': 20,
        'retained_source_bytes_and_order_verified': True,
        'unchanged_holdouts_and_backgrounds': True,
        'input_receipts': input_receipts,
        'canonical_preflight': 'pending_in_trainers',
        'inherited_full_verification': prior['verification'],
    }
    from cohort_composition import summarize

    summarize(output / 'wide-multi-s42.json', output / 'cohort-composition.json')
    composition = read(output / 'cohort-composition.json')
    observed = Counter()
    for item in composition['populations']:
        observed[item['split'] + ':' + item['endpoint']] += (
            item['n'] if item['population'] == 'original' else 0
        )
    assert dict(observed) == p['physical_observations']
    full_counts = Counter()
    for item in composition['populations']:
        full_counts[item['split'] + ':' + item['endpoint'] + ':' + item['mhc_class']] += item['n']
    expected_counts = Counter()
    for key, n in p['counts'].items():
        expected_counts[':'.join(key.split(':')[:3])] += n
    assert full_counts == expected_counts
    write(
        output / 'cohort-composition-verification.json',
        dict(
            status='passed',
            source_sha256=composition['source_sha256'],
            result_sha256=sha(output / 'cohort-composition.json'),
            checked='source hashes, original split/endpoint and total split/endpoint/class counts',
            recorded_unix=time.time(),
        ),
    )
    p['source_review'] = dict(
        revision=REVISION,
        policy_sha256=policy_sha,
        removed_rows=20,
        reasons=dict(removed_counts),
        runtime_source=pins['git_commit'],
    )
    p['finished_unix'] = time.time()
    write(output / 'prepared.json', p)
    result = {
        **p,
        'prepared_record_sha256': sha(output / 'prepared.json'),
        'verification': verification,
        'review_metadata': {
            name: read(output / name)
            for name in (
                'study-selection.json',
                'endpoint-support.json',
                'preflight.json',
                'wide-multi-s42.json',
                'wide-multi-s43.json',
                'source-review-dispositions.json',
                'cohort-composition.json',
                'cohort-composition-verification.json',
            )
        },
    }
    durable.mkdir(parents=True, exist_ok=False)
    shutil.copytree(output, durable / 'prepared')
    result['storage'] = dict(
        revision_directory=str(durable),
        started_unix=started,
        finished_unix=time.time(),
        nonpreemptible=True,
    )
    write(durable / 'preparation-result.json', result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    path = F / 'reproduce/prepare-call.json'
    assert not path.exists(), (
        'archive stopped attempt first; recover an existing revision by saved call'
    )
    shutil.copyfile(__file__, F / 'reproduce/source/revise_cohort.py')
    write(
        F / 'reproduce/cohort-revision-launch.json',
        dict(
            argv=['modal', 'run', '--detach', str(Path(__file__).resolve())],
            revision=REVISION,
            cpu=8,
            memory_mib=65536,
            nonpreemptible=True,
            gpu=None,
            source_sha256=sha(__file__),
            runtime_pins_sha256=sha(U / 'runtime-pins.json'),
        ),
    )
    call = revise.spawn()
    write(
        path,
        dict(
            call_id=call.object_id,
            revision=REVISION,
            started_unix=time.time(),
            source_sha256=sha(__file__),
        ),
    )
    result = call.get()
    write(F / 'results/prepared.json', result)
    write(F / 'results/partition-verification.json', result['verification'])
    for name, value in result['review_metadata'].items():
        write(F / 'results' / name, value)
    print('Prepared reviewed cohort', result['source_review'], flush=True)
