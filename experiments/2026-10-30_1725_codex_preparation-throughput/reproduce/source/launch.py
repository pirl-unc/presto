"""CPU-only paired runtime study on exact prefixes of the active cohort."""

from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import modal

FID = '2026-10-30_1725_codex_preparation-throughput'
PARENT = '2026-10-29_2303_codex_assay-measurement-layer'
app = modal.App('presto-preparation-throughput-20261030')
volume = modal.Volume.from_name('presto-checkpoints')
image = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install(
        'torch==2.14.1',
        'numpy==2.5.4',
        'pyyaml==6.0.3',
        'mhcgnomes==3.64.6',
        'mhcseqs==2.6.13',
        'tqdm==4.70.1',
        'psutil==7.2.2',
    )
    .env(
        {
            'OMP_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4',
            'PYTHONDONTWRITEBYTECODE': '1',
        }
    )
)
if modal.is_local():
    family = Path(__file__).resolve().parents[1]
    launch = json.loads((family / 'reproduce/launch.json').read_text())
    for variant, snapshot in launch['runtimes'].items():
        for relative, expected in snapshot['sha256'].items():
            assert (
                hashlib.sha256((Path(snapshot['path']) / relative).read_bytes()).hexdigest()
                == expected
            )
        image = image.add_local_dir(snapshot['path'], remote_path=f'/runtime/{variant}/presto')
    image = image.add_local_dir(family / 'code', remote_path='/bench')
    image = image.add_local_dir(family / 'reproduce', remote_path='/bundle')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@app.function(image=image, cpu=8, memory=65536, timeout=7200, volumes={'/volume': volume})
def benchmark():
    sys.path.insert(0, '/runtime/candidate')
    from presto.data.assay_measurement import observation_track
    from presto.data.modular_sources import normalized_observation
    from presto.models.assay_measurement import AFFINITY_ENDPOINTS

    root = Path('/study')
    root.mkdir()
    durable = Path('/volume') / FID
    if durable.exists():
        raise RuntimeError('refusing to overwrite an existing runtime experiment')
    parent = json.loads(Path('/bundle/parent-manifest.json').read_text())
    endpoints, classes, tracks, strata, sources = set(), set(), {}, Counter(), []
    inputs = []
    for name in ('hitlist-v4-iedb', 'hitlist_ligand_background', 'hitlist_native_ms'):
        original = next(
            s for s in parent['sources'] if Path(s['path']).name == name + '-train.jsonl'
        )
        origin = Path('/volume') / PARENT / 'prepared' / Path(original['path']).name
        path = root / origin.name
        with origin.open('rb') as stream:
            path.write_bytes(b''.join(itertools.islice(stream, 5000)))
        source = {**original, 'path': str(path), 'sha256': sha(path)}
        sources.append(source)
        count = 0
        for line in path.open():
            row = normalized_observation(json.loads(line), evidence_policy=source['label_policy'])
            assert row.split == 'train'
            endpoints.add(row.endpoint)
            cls = next(iter({m.class_override or m.mhc_class for m in row.mhc}))
            classes.add(cls)
            strata[(row.endpoint, cls)] += 1
            track = observation_track(row)
            if track and row.endpoint in AFFINITY_ENDPOINTS:
                tracks[track.key] = asdict(track)
            count += 1
        inputs.append(
            {
                'volume_path': str(origin),
                'full_source_sha256_declared': original['sha256'],
                'prefix_sha256': source['sha256'],
                'rows': count,
                'bytes': path.stat().st_size,
            }
        )
    manifest = {
        'schema': parent['schema'],
        'sources': sources,
        'model': {
            **parent['model'],
            'endpoints': sorted(endpoints),
            'mhc_classes': sorted(classes),
            'assay_mode': 'factorized',
            'assay_tracks': [tracks[k] for k in sorted(tracks)],
        },
        'training': {
            **parent['training'],
            'test_policy': 'no_test_training_diagnostic',
            'no_test_reason': 'CPU preparation benchmark only; no optimization or prediction',
            'coverage_policy': 'diagnostic',
            'coverage_exception_reason': 'CPU preparation benchmark; no optimization',
            'stratum_weights': [
                s for s in parent['training']['stratum_weights'] if tuple(s['values']) in strata
            ],
        },
    }
    write(root / 'manifest.json', manifest)
    write(root / 'inputs.json', inputs)
    results = []
    for repeat in range(3):
        for variant in ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline'):
            destination = root / f'{variant}-{repeat}.json'
            command = [
                sys.executable,
                '/bench/benchmark.py',
                '--runtime',
                f'/runtime/{variant}',
                '--manifest',
                str(root / 'manifest.json'),
                '--output',
                str(destination),
                '--condition',
                variant,
            ]
            log = root / f'{variant}-{repeat}.log'
            with log.open('w') as output:
                subprocess.run(
                    command,
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONPATH': f'/runtime/{variant}'},
                )
            result = json.loads(destination.read_text())
            if results:
                assert result['invariants'] == results[0]['invariants'], 'preparation parity failed'
            results.append(result)
            print(variant, repeat, result['total_seconds'], flush=True)
    timings = {}
    for phase in (*results[0]['phases'], 'total'):
        medians = {
            v: statistics.median(
                r['total_seconds'] if phase == 'total' else r['phases'][phase]['seconds']
                for r in results
                if r['condition'] == v
            )
            for v in ('baseline', 'candidate')
        }
        timings[phase] = {**medians, 'speedup': medians['baseline'] / medians['candidate']}
    summary = {
        'timings': timings,
        'parity': True,
        'rows': sum(s['rows'] for s in inputs),
        'bytes': sum(s['bytes'] for s in inputs),
        'requested_cpu': 8,
        'requested_memory_mib': 65536,
        'requested_gpu': None,
        'hardware': results[0]['hardware'],
        'finished_unix': time.time(),
    }
    write(root / 'summary.json', summary)
    shutil.copytree(root, durable)
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    result = benchmark.remote()
    write(Path(__file__).resolve().parents[1] / 'results/summary.json', result)
    print(json.dumps(result, indent=2))
