"""Canonical fresh Hitlist index plus outcome-independent publication inventory."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import time
import os
import shutil
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-temporal-source-index-20261028')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12').uv_pip_install('hitlist==2.0.40')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image,
    cpu=8,
    memory=98304,
    ephemeral_disk=262144,
    timeout=14400,
    max_containers=1,
    volumes={'/runs': volume},
)
def build():
    import pyarrow.csv as pc
    import pyarrow.parquet as pq
    from hitlist.downloads import set_data_dir
    from hitlist.builder import build_observations

    root = Path('/runs') / FID / 'source'
    receipt = json.loads((root / 'acquisition.json').read_text())
    assert receipt['status'] == 'downloaded'
    set_data_dir(root)
    result = {
        'started_unix': time.time(),
        'status': 'building',
        'hitlist': '2.0.40',
        'requested_gpu': None,
        'cpu': 16,
        'memory_mib': 98304,
        'sources': receipt['sources'],
    }

    def save():
        (root / 'build.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        volume.commit()

    if (root / 'build.json').exists():
        old = json.loads((root / 'build.json').read_text())
        if old.get('status') == 'failed':
            write_to = root / f"build-failed-{int(old['updated_unix'])}.json"
            write_to.write_text(json.dumps(old, indent=2) + '\n')
    result['environment_overrides'] = {
        name: os.environ[name]
        for name in (
            'HITLIST_PROVENANCE_MAX_GB',
            'HITLIST_PROVENANCE_MIN_FREE_GB',
            'TQDM_MININTERVAL',
        )
    }
    result['requested_ephemeral_disk_mib'] = 262144
    result['scratch_disk_start'] = dict(zip(('total', 'used', 'free'), shutil.disk_usage('/tmp')))
    save()
    modal.Dict.from_name('presto-cross-class-live').put(
        FID + ':preparation',
        {
            'status': 'source_index_retry',
            'stage': 'Rebuilding full provenance with a 128 GiB scratch database limit and 256 GiB requested disk; source downloads preserved',
            'updated_unix': time.time(),
            'counts': {},
        },
    )
    try:
        refs = {}
        for source, spec in receipt['sources'].items():
            names = [f'{a} | {b}' for a, b in zip(spec['header1'], spec['header2'], strict=True)]
            assert len(set(names)) == len(names)
            wanted = [name for name in names if name.startswith('Reference | ')]
            table = pc.open_csv(
                spec['path'],
                read_options=pc.ReadOptions(column_names=names, skip_rows=2, block_size=1 << 24),
                parse_options=pc.ParseOptions(newlines_in_values=True),
                convert_options=pc.ConvertOptions(
                    include_columns=wanted, column_types={name: 'string' for name in wanted}
                ),
            )
            for batch in table:
                for row in batch.to_pylist():
                    key = json.dumps(row, sort_keys=True)
                    if key not in refs:
                        refs[key] = {'source': source, 'metadata': row, 'source_rows': 0}
                    refs[key]['source_rows'] += 1
            print(source, 'reference inventory', len(refs), flush=True)
        with (root / 'references.jsonl').open('w') as stream:
            for row in refs.values():
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        result['reference_inventory'] = {
            'path': str(root / 'references.jsonl'),
            'sha256': sha(root / 'references.jsonl'),
            'records': len(refs),
        }
        save()
        path = build_observations(
            with_flanking=False, build_mappings=False, fetch_missing_proteomes=False
        )
        result['dataset_root'] = str(path.parent)
        result['files'] = {}
        inventory = []
        fields = [
            'pmid',
            'reference_iri',
            'reference_title',
            'mhc_class',
            'mhc_species',
            'assay_method',
            'response_measured',
            'measurement_units',
            'source',
            'ms_outcome_status',
            'mhc_allele_provenance',
            'restriction_evidence',
        ]
        for name in ('binding.parquet', 'other_assays.parquet', 'observations.parquet'):
            path = path.parent / name
            pf = pq.ParquetFile(path)
            counts = Counter()
            for batch in pf.iter_batches(columns=fields, batch_size=65536):
                columns = [batch.column(i).to_pylist() for i in range(len(fields))]
                counts.update(zip(*columns, strict=True))
            result['files'][name] = {
                'path': str(path),
                'size_bytes': path.stat().st_size,
                'sha256': sha(path),
                'rows': pf.metadata.num_rows,
            }
            for values, n in counts.items():
                inventory.append({'index': name, **dict(zip(fields, values, strict=True)), 'n': n})
            print(name, pf.metadata.num_rows, 'rows', flush=True)
        (root / 'inventory.json').write_text(
            json.dumps(inventory, indent=2, allow_nan=False) + '\n'
        )
        result.update(
            status='built',
            inventory_path=str(root / 'inventory.json'),
            inventory_sha256=sha(root / 'inventory.json'),
        )
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['updated_unix'] = time.time()
        save()
    return result


@app.local_entrypoint()
def main():
    family = Path(__file__).resolve().parents[1]
    (family / 'reproduce/build-launch.json').write_text(
        json.dumps(
            {
                'argv': ['modal', 'run', '--detach', str(Path(__file__).resolve())],
                'source_sha256': sha(Path(__file__)),
                'started_unix': time.time(),
                'requested_gpu': None,
                'cpu': 16,
                'memory_mib': 98304,
                'requirements': ['hitlist==2.0.40'],
            },
            indent=2,
        )
        + '\n'
    )
    call = build.spawn()
    (family / 'reproduce/build-call.json').write_text(
        json.dumps({'call_id': call.object_id}, indent=2) + '\n'
    )
    result = call.get()
    (family / 'results/build.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], flush=True)
