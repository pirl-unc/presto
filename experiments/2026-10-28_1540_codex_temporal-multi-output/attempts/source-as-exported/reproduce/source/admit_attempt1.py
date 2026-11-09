"""Prepare source-traceable candidates with canonical adapters; never fit a model."""

from pathlib import Path
from collections import Counter
from dataclasses import replace
import csv, gzip, hashlib, json, sqlite3, sys, time
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'
LOCAL = Path(__file__).resolve().parents[1]
UPLOAD = LOCAL.parents[1] / 'artifacts' / FID / 'upload'
app = modal.App('presto-temporal-admission-20261028')
volume = modal.Volume.from_name('presto-checkpoints')
live = modal.Dict.from_name('presto-cross-class-live', create_if_missing=True)
requirements = [
    'hitlist==2.0.40',
    'torch==2.14.1',
    'mhcseqs==2.6.13',
    'numpy==2.5.4',
    'psutil==7.2.2',
]
image = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install(*requirements)
    .env(
        {
            'OMP_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4',
            'TQDM_DISABLE': '1',
        }
    )
)
if modal.is_local():
    image = (
        image.add_local_dir(UPLOAD / 'presto', remote_path='/runtime/presto')
        .add_local_file(UPLOAD / 'runtime-pins.json', remote_path='/bundle/runtime-pins.json')
        .add_local_file(UPLOAD / 'mhc-full-seqs.csv', remote_path='/bundle/mhc-full-seqs.csv')
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@app.function(
    image=image, cpu=8, memory=98304, timeout=14400, max_containers=1, volumes={'/runs': volume}
)
def admit(build_call_id):
    # A frozen dependency, not a second index build.
    build = modal.FunctionCall.from_id(build_call_id).get()
    assert build['status'] == 'built'
    volume.reload()
    sys.path.insert(0, '/runtime')
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hitlist.downloads import set_data_dir
    from hitlist.export import generate_training_table
    from presto.data.modular_hitlist import (
        HumanMHCCatalog,
        AdmissionError,
        purified_assay_observation,
    )
    from presto.data.modular_hitlist_ms import ms_observation, study_factor_vocabulary
    from presto.data.modular_lineage import measurement_key, semantic_fingerprint

    pins = json.loads(Path('/bundle/runtime-pins.json').read_text())
    for name, digest in pins['files'].items():
        assert sha(Path('/runtime/presto') / name) == digest, name
    assert sha('/bundle/mhc-full-seqs.csv') == pins['catalog']['sha256']
    source = Path('/runs') / FID / 'source'
    set_data_dir(source)
    root = Path('/runs') / FID / 'admission'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'admitted.sqlite').exists():
        raise ValueError('Already attempted: preserve source admission and use explicit recovery')
    result = {
        'status': 'admitting',
        'started_unix': time.time(),
        'runtime_pins': pins,
        'build': build,
        'counts': {},
        'requested_gpu': None,
        'cpu': 8,
        'memory_mib': 98304,
    }

    def status(stage):
        result.update(stage=stage, updated_unix=time.time(), counts=dict(counts))
        write(root / 'status.json', result)
        live.put(
            FID + ':preparation',
            {
                'stage': stage,
                'status': result['status'],
                'counts': dict(counts),
                'updated_unix': time.time(),
            },
        )

    counts = Counter()
    with open('/bundle/mhc-full-seqs.csv') as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    db = sqlite3.connect(root / 'admitted.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE rows (id TEXT PRIMARY KEY, lineage TEXT UNIQUE, peptide TEXT, endpoint TEXT, mhc_class TEXT, study TEXT, provenance_id TEXT, semantic TEXT, excluded INTEGER DEFAULT 0, data TEXT);
    CREATE TABLE aliases (id TEXT, study TEXT, PRIMARY KEY(id,study));
    CREATE TABLE source_studies (provenance_id TEXT,study TEXT,PRIMARY KEY(provenance_id,study));
    CREATE TABLE positive_peptides (peptide TEXT PRIMARY KEY);''')
    disposition = gzip.open(root / 'dispositions.jsonl.gz', 'wt')

    def consume(raw, source_name, ordinal, digest, adapter):
        counts['scanned:' + source_name] += 1
        try:
            row = adapter(
                raw,
                catalog,
                source_sha256=digest,
                ordinal=ordinal,
                dra_reference=FID + ':human_dr_dra0101_v1',
            )
            pmid = str(raw.get('pmid') or '').strip()
            ref = str(raw.get('reference_iri') or '').strip()
            if not pmid and not ref:
                raise AdmissionError('missing_study_identity', deferred=True)
            study = 'pmid:' + pmid if pmid else 'reference:' + ref
            row = replace(
                row,
                observation_id='temporal:' + source_name + ':' + str(ordinal),
                provenance={
                    **row.provenance,
                    'study_id': study,
                    'source_file': source_name,
                    'source_contributor_id': raw.get('provenance_id'),
                },
            )
            key = json.dumps(measurement_key(row), separators=(',', ':'))
            semantic = semantic_fingerprint(row)
            existing = db.execute('SELECT id,semantic FROM rows WHERE lineage=?', (key,)).fetchone()
            if existing:
                db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?)', (existing[0], study))
                reason = (
                    'duplicate_original_outcome'
                    if existing[1] == semantic
                    else 'conflicting_original_outcome'
                )
                if reason.startswith('conflicting'):
                    db.execute('UPDATE rows SET excluded=1 WHERE id=?', (existing[0],))
                raise AdmissionError(reason, deferred=reason.startswith('conflicting'))
            db.execute(
                'INSERT INTO rows(id,lineage,peptide,endpoint,mhc_class,study,provenance_id,semantic,data) VALUES (?,?,?,?,?,?,?,?,?)',
                (
                    row.observation_id,
                    key,
                    row.peptide,
                    row.endpoint,
                    row.mhc[0].mhc_class,
                    study,
                    raw.get('provenance_id') or '',
                    semantic,
                    json.dumps(row.to_dict(), separators=(',', ':'), allow_nan=False),
                ),
            )
            db.execute('INSERT INTO aliases VALUES (?,?)', (row.observation_id, study))
            counts['admitted:' + row.endpoint + ':' + row.mhc[0].mhc_class] += 1
            counts['declared_dra_reference'] += any(
                m.provenance.get('partner_imputation') for m in row.mhc
            )
        except AdmissionError as exc:
            counts[exc.disposition + ':' + exc.reason] += 1
            disposition.write(
                json.dumps(
                    {
                        'source': source_name,
                        'ordinal': ordinal,
                        'disposition': exc.disposition,
                        'reason': exc.reason,
                    }
                )
                + '\n'
            )

    try:
        for name in ('binding.parquet', 'other_assays.parquet'):
            spec = build['files'][name]
            path = Path(spec['path'])
            assert sha(path) == spec['sha256']
            ordinal = 0
            status('admitting ' + name)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                for raw in batch.to_pylist():
                    consume(raw, name, ordinal, spec['sha256'], purified_assay_observation)
                    ordinal += 1
                db.commit()
                status('admitting ' + name)
        db.commit()
        status('indexing all known positive ligand sequences')
        for batch in pq.ParquetFile(build['files']['observations.parquet']['path']).iter_batches(
            columns=['peptide'], batch_size=65536
        ):
            db.executemany(
                'INSERT OR IGNORE INTO positive_peptides VALUES (?)',
                [(p,) for p in batch.column(0).to_pylist() if p],
            )
            db.commit()
        status('exporting canonical human MS evidence')
        # The upstream export enriches candidate typing and source-condition scope.
        frame = generate_training_table(
            include_evidence='ms',
            species='Homo sapiens',
            exclude_chimeric=True,
            map_source_proteins=False,
            with_peptide_origin=False,
        )
        export = root / 'canonical-ms-export.parquet'
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), export, compression='zstd'
        )
        del frame
        digest = sha(export)
        pf = pq.ParquetFile(export)
        factors = {}
        # Vocabulary contains declared source factors, not labels; batch unions are exact.
        for batch in pf.iter_batches(batch_size=8192):
            raw_rows = batch.to_pylist()
            for pmid, fields in study_factor_vocabulary(raw_rows).items():
                for field, values in fields.items():
                    factors.setdefault(pmid, {}).setdefault(field, set()).update(values)
            db.executemany(
                'INSERT OR IGNORE INTO positive_peptides VALUES (?)',
                [(r['peptide'],) for r in raw_rows if r.get('peptide')],
            )
        db.commit()
        factors = {
            p: {k: tuple(sorted(v)) for k, v in fields.items()} for p, fields in factors.items()
        }
        ordinal = 0

        def ms_adapter(*args, **kwargs):
            return ms_observation(*args, study_factors=factors, **kwargs)

        for batch in pf.iter_batches(batch_size=8192):
            for raw in batch.to_pylist():
                consume(raw, 'canonical-ms-export.parquet', ordinal, digest, ms_adapter)
                ordinal += 1
            db.commit()
            status('admitting canonical MS evidence')
        status('joining original contributor study aliases')
        contributors = Path(build['dataset_root']) / 'observation_contributors.parquet'
        for batch in pq.ParquetFile(contributors).iter_batches(
            columns=['provenance_id', 'original_pmid'], batch_size=65536
        ):
            pairs = []
            for pid, pmid in zip(*(batch.column(i).to_pylist() for i in range(2)), strict=True):
                if pid and pmid:
                    pairs.append((pid, 'pmid:' + str(pmid).strip()))
            db.executemany('INSERT OR IGNORE INTO source_studies VALUES (?,?)', pairs)
            db.commit()
        db.executescript(
            'CREATE INDEX rows_provenance ON rows(provenance_id); CREATE INDEX rows_study ON rows(study); CREATE INDEX rows_peptide ON rows(peptide);'
        )
        census = [
            {'study': s, 'endpoint': e, 'mhc_class': c, 'n': n}
            for s, e, c, n in db.execute(
                'SELECT study,endpoint,mhc_class,COUNT(*) FROM rows WHERE excluded=0 GROUP BY study,endpoint,mhc_class'
            )
        ]
        write(root / 'study-census.json', census)
        result.update(
            status='admitted',
            admitted_rows=db.execute('SELECT COUNT(*) FROM rows WHERE excluded=0').fetchone()[0],
            source_dispositions=dict(counts),
            study_census=census,
            canonical_ms_export={
                'path': str(export),
                'sha256': digest,
                'rows': pf.metadata.num_rows,
            },
            source_contributors={'path': str(contributors), 'sha256': sha(contributors)},
            database_path=str(root / 'admitted.sqlite'),
        )
        write(root / 'used-catalog-aliases.json', catalog.aliases())
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        disposition.close()
        db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.close()
        status(result['status'])
        volume.commit()
    return result


@app.local_entrypoint()
def main():
    f = Path(__file__).resolve().parents[1]
    call_id = json.loads((f / 'reproduce/build-call.json').read_text())['call_id']
    write(
        f / 'reproduce/admission-launch.json',
        {
            'argv': ['modal', 'run', '--detach', str(Path(__file__).resolve())],
            'source_sha256': sha(__file__),
            'dependency_call_id': call_id,
            'requirements': requirements,
            'requested_gpu': None,
            'cpu': 8,
            'memory_mib': 98304,
            'started_unix': time.time(),
        },
    )
    print('Waiting for frozen canonical index build', call_id, flush=True)
    modal.FunctionCall.from_id(call_id).get()
    call = admit.spawn(call_id)
    write(f / 'reproduce/admission-call.json', {'call_id': call.object_id})
    result = call.get()
    write(f / 'results/admission.json', result)
    print(result['status'], result.get('admitted_rows'), flush=True)
