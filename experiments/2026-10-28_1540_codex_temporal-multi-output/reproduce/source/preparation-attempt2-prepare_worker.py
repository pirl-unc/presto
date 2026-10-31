"""Freeze paper membership, backgrounds, manifests and common-trainer preflight."""

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import time

from partition import census_components, select

FID = '2026-10-28_1540_codex_temporal-multi-output'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def weights(counts):
    # Frequencies, not multiplicative loss weights. Cover every row at least once.
    families = {
        'elution_observation': 0.60,
        'ic50': 0.20,
        'physical_other': 0.15,
        'qualitative_binding': 0.05,
    }

    def family(endpoint):
        return endpoint if endpoint in families else 'physical_other'

    totals = Counter()
    for (endpoint, _cls), n in counts.items():
        totals[family(endpoint)] += math.sqrt(n)
    return {
        key: families[family(key[0])] * math.sqrt(n) / totals[family(key[0])]
        for key, n in counts.items()
    }


def prepare(root, status, *, bundle='/bundle', runtime='/runtime/presto'):
    bundle, runtime = Path(bundle), Path(runtime)
    sys.path.insert(0, str(runtime.parent))
    from hitlist.proteome import ProteomeIndex
    from presto.data.modular import Observation, endpoint_spec, transformed_target
    from presto.data.modular_background import LigandBackgroundSampler
    from presto.data.modular_sources import ObservationIndex, MANIFEST_VERSION
    from presto.training.modular_sampling import weighted_quotas
    from presto.training.modular_run import preflight

    root = Path(root)
    output = root / 'prepared'
    output.mkdir(exist_ok=False)
    pins = json.loads((bundle / 'runtime-pins.json').read_text())
    for relative, digest in pins['files'].items():
        assert sha(runtime / relative) == digest, relative
    db = sqlite3.connect(f'file:{root}/admission/admitted.sqlite?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=MEMORY')
    status('forming whole-study contributor components')
    components, studies = census_components(db, root / 'source/references.jsonl')
    assignments, selection = select(studies)
    write(output / 'study-selection.json', selection)
    study_split = {name: assignments[r] for r, spec in studies.items() for name in spec['aliases']}
    study_group = {name: spec['id'] for spec in studies.values() for name in spec['aliases']}
    group_aliases = {spec['id']: spec['aliases'] for spec in studies.values()}
    group_years = {spec['id']: spec['years'] for spec in studies.values()}
    priority = {'train': 0, 'validation': 1, 'test': 2}
    peptide_splits = {}
    status('freezing exact-peptide embargoes')
    for peptide, study in db.execute('SELECT peptide,study FROM rows WHERE excluded=0'):
        split = study_split[study]
        if priority[split] > priority.get(peptide_splits.get(peptide), -1):
            peptide_splits[peptide] = split
    # Scaler support is examined only after membership selection is frozen.
    exact_values = defaultdict(set)
    train_endpoints = Counter()
    for peptide, endpoint, study, qualifier, value, unit in db.execute(
        "SELECT peptide,endpoint,study,json_extract(data,'$.qualifier'),"
        "json_extract(data,'$.value'),json_extract(data,'$.unit') FROM rows WHERE excluded=0"
    ):
        if study_split[study] == 'train' and peptide_splits[peptide] == 'train':
            train_endpoints[endpoint] += 1
            if qualifier == '=' and len(exact_values[endpoint]) < 2:
                exact_values[endpoint].add(transformed_target(endpoint, value, unit, qualifier))
    endpoints = sorted(
        e for e in train_endpoints if endpoint_spec(e).binary or len(exact_values[e]) >= 2
    )
    unsupported = sorted(set(train_endpoints) - set(endpoints))
    write(
        output / 'endpoint-support.json',
        dict(
            train_counts=dict(train_endpoints),
            allocated=endpoints,
            unsupported=unsupported,
            rule='nonbinary outputs require two distinct exact TRAIN values',
        ),
    )
    status('loading pinned reference proteome for length-matched comparisons')
    fasta = output / 'human-ensembl112.pep.fa'
    assert sha(bundle / 'human-ensembl112.pep.fa.gz') == pins['proteome']['sha256']
    with gzip.open(bundle / 'human-ensembl112.pep.fa.gz', 'rb') as source, fasta.open('wb') as dest:
        shutil.copyfileobj(source, dest)
    proteome = ProteomeIndex.from_fasta(str(fasta))
    positive_peptides = {p for (p,) in db.execute('SELECT peptide FROM positive_peptides')}
    sampler = LigandBackgroundSampler(
        proteome,
        seed=20261028,
        reference=FID + ':length-matched-background-v1',
        positive_peptides=positive_peptides,
        peptide_splits=peptide_splits,
    )
    sources, handles = {}, {}
    counts, dispositions = Counter(), Counter()
    background_counts = Counter()
    train_strata = Counter()
    cells = Counter()
    negative_studies = Counter()
    admission_census = Counter()
    excluded = gzip.open(output / 'partition-dispositions.jsonl.gz', 'wt')
    membership = gzip.open(output / 'membership.jsonl.gz', 'wt')

    def emit(row):
        key = (row.source, row.split)
        if key not in handles:
            p = output / (row.source + '-' + row.split + '.jsonl')
            handles[key] = p.open('x')
            sources[key] = p
        handles[key].write(json.dumps(row.to_dict(), separators=(',', ':'), allow_nan=False) + '\n')
        membership.write(
            json.dumps(
                dict(
                    id=row.observation_id,
                    peptide_sha256=hashlib.sha256(row.peptide.encode()).hexdigest(),
                    endpoint=row.endpoint,
                    group=row.group_id,
                    split=row.split,
                    parent=row.provenance.get('parent_observation_id'),
                )
            )
            + '\n'
        )
        cls = row.mhc[0].mhc_class
        counts[(row.split, row.endpoint, cls, row.qualifier)] += 1
        if row.split == 'train':
            train_strata[(row.endpoint, cls)] += 1
        raw = row.provenance.get('raw_record', {})
        if row.source != 'hitlist_ligand_background':
            for field in ('assay_method', 'mhc_species', 'source_organism', 'host', 'species'):
                cells[(row.split, row.endpoint, field, str(raw.get(field) or 'unknown'))] += 1
            cells[(row.split, row.endpoint, 'peptide_length', str(len(row.peptide)))] += 1
            cells[(row.split, row.endpoint, 'candidate_count', str(len(row.mhc)))] += 1
            for mhc in row.mhc:
                for allele in mhc.provenance.get('reported_candidates', []):
                    cells[(row.split, row.endpoint, 'reported_candidate_allele', allele)] += 1
        if row.provenance.get('label_kind') == 'reported_nondetection':
            negative_studies[(row.split, row.provenance['study_id'])] += 1

    for split in ('test', 'validation', 'train'):
        status('materializing ' + split + ' observations and backgrounds')
        emitted = 0
        # Stable observation-ID order, same backgrounds for both initialization seeds.
        for identifier, peptide, endpoint, study, payload in db.execute(
            'SELECT id,peptide,endpoint,study,data FROM rows WHERE excluded=0 ORDER BY id'
        ):
            if study_split[study] != split:
                continue
            reason = (
                'heldout_peptide_embargo'
                if peptide_splits[peptide] != split
                else 'output_without_train_scale'
                if endpoint not in endpoints
                else None
            )
            if reason:
                dispositions[(split, reason, endpoint)] += 1
                excluded.write(json.dumps(dict(id=identifier, split=split, reason=reason)) + '\n')
                continue
            row = Observation.from_dict(json.loads(payload))
            group = study_group[study]
            row = replace(
                row,
                split=split,
                group_id=group,
                provenance={
                    **row.provenance,
                    'group_aliases': group_aliases[group],
                    'study_component_years': group_years[group],
                    'evaluation_era': '2026'
                    if group_years[group] and min(group_years[group]) >= 2026
                    else 'older_or_mixed',
                    'partition_policy': FID + ':whole-study-peptide-embargo-v1',
                },
            )
            if endpoint == 'elution_observation' and row.value == 1:
                backgrounds = []
                for replicate in range(1 if split == 'train' else 5):
                    try:
                        backgrounds.append(sampler.sample(row, replicate=replicate))
                    except ValueError as exc:
                        if str(exc) not in {
                            'no protein supports the requested background length',
                            'background population exhausted after positive and split exclusion',
                        }:
                            raise
                        dispositions[(split, 'background_exhausted', str(exc))] += 1
                        excluded.write(
                            json.dumps(
                                dict(
                                    id=row.observation_id,
                                    split=split,
                                    reason='background_exhausted',
                                    replicate=replicate,
                                    detail=str(exc),
                                )
                            )
                            + '\n'
                        )
                        break
                population = (
                    'synthetic_' + backgrounds[0].provenance['background_kind']
                    if backgrounds
                    else 'no_background'
                )
                row = replace(
                    row, provenance={**row.provenance, 'evaluation_population': population}
                )
                emit(row)
                for bg in backgrounds:
                    bg = replace(
                        bg,
                        provenance={
                            **bg.provenance,
                            'evaluation_population': population,
                            'evaluation_era': row.provenance['evaluation_era'],
                        },
                    )
                    emit(bg)
                    background_counts[(split, population)] += 1
            else:
                if endpoint == 'elution_observation':
                    row = replace(
                        row,
                        provenance={
                            **row.provenance,
                            'evaluation_population': 'source_reported_nondetection',
                        },
                    )
                emit(row)
            admission_census[(split, endpoint)] += 1
            emitted += 1
            if emitted % 20000 == 0:
                status(
                    'materializing ' + split,
                    dict(rows=emitted, backgrounds=sum(background_counts.values())),
                )
    for stream in handles.values():
        stream.close()
    excluded.close()
    membership.close()
    db.close()
    del proteome, positive_peptides, peptide_splits
    sampling_weights = weights(train_strata)
    quotas = weighted_quotas(train_strata, sampling_weights)
    batches = math.ceil(sum(quotas.values()) / 128)
    steps = max(6000, batches)
    config = dict(
        endpoints=endpoints,
        mhc_classes=['I', 'II'],
        d_model=192,
        n_layers=3,
        n_heads=6,
        d_ff=768,
        dropout=0.1,
        source_context=False,
    )
    source_specs = [
        dict(
            path=str(path),
            sha256=sha(path),
            source=source,
            population=(
                'Human catalog-resolved MHC; public IEDB/CEDAR canonical Hitlist admission; '
                'host and antigen species retained separately'
            ),
            label_policy=(
                'reported literal assay endpoints and qualifiers; declared reference DRA '
                'only; generated unobserved ligands explicitly marked'
            ),
            split_policy=selection['policy'],
        )
        for (source, split), path in sorted(sources.items())
    ]
    manifests = {}
    for seed in (42, 43):
        condition = 'wide-multi-s' + str(seed)
        manifest = dict(
            schema=MANIFEST_VERSION,
            model=config,
            sources=source_specs,
            training=dict(
                seed=seed,
                batch_size=128,
                max_epochs=math.ceil(steps / batches),
                max_steps=steps,
                learning_rate=3e-4,
                weight_decay=1e-4,
                identity_weight=0.0,
                patience=100000,
                min_epochs=100000,
                checkpoint_every_steps=500,
                validation_every_steps=1000,
                selection_reduction='endpoint_mean',
                train_evaluation_max_rows=16384,
                balance_by=['endpoint', 'mhc_class'],
                stratum_weights=[
                    dict(values=list(key), weight=value)
                    for key, value in sorted(sampling_weights.items())
                ],
            ),
        )
        path = output / (condition + '.json')
        write(path, manifest)
        manifests[condition] = dict(path=str(path), sha256=sha(path), seed=seed, max_steps=steps)
    status('canonical observation-index and common-trainer preflight')
    index = ObservationIndex(manifests['wide-multi-s42']['path'])
    _, training, scales, admission = preflight(index)
    write(
        output / 'preflight.json',
        dict(
            status='passed',
            training=training,
            admission=admission,
            scales={k: vars(v) for k, v in scales.items()},
        ),
    )
    result = dict(
        status='prepared',
        family=FID,
        manifests=manifests,
        model=config,
        counts={':'.join(k): n for k, n in sorted(counts.items())},
        dispositions={':'.join(k): n for k, n in sorted(dispositions.items())},
        background_counts={':'.join(k): n for k, n in sorted(background_counts.items())},
        train_strata={':'.join(k): n for k, n in sorted(train_strata.items())},
        epoch_quotas={':'.join(k): n for k, n in sorted(quotas.items())},
        steps=steps,
        weighted_epoch_examples=sum(quotas.values()),
        minimum_unique_row_passes=1,
        physical_observations={':'.join(k): n for k, n in sorted(admission_census.items())},
        reported_negative_studies={':'.join(k): n for k, n in sorted(negative_studies.items())},
        background_contract=dict(
            train_ratio=1,
            evaluation_ratio=5,
            seed=20261028,
            with_replacement=True,
            positive_blacklist=(
                'all canonical known positive peptides, including held-out sources; '
                'conservative exclusion only'
            ),
            proteome=pins['proteome'],
            primary_evaluation='same-protein; fallback proteome reported separately',
        ),
        finished_unix=time.time(),
    )
    write(
        output / 'strata.json',
        [
            dict(split=k[0], endpoint=k[1], field=k[2], value=k[3], n=n)
            for k, n in sorted(cells.items())
        ],
    )
    write(output / 'prepared.json', result)
    return result
