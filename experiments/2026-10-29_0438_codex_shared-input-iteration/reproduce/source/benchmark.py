"""Warm-cache CPU reader comparison; no model, optimizer or held-out inference."""

from pathlib import Path
import csv
import gc
import hashlib
import json
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
sys.path.insert(0, str(ROOT.parent))
from presto.data.modular_sources import ObservationIndex, MANIFEST_VERSION  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def original(index):
    for i in range(len(index)):
        yield index[i]


def main():
    assert not (F / 'results/summary.json').exists(), 'completed benchmark is immutable'
    capture = json.loads((F / 'results/capture.json').read_text())
    A = Path(capture['local_artifacts'])
    sources = []
    for source in capture['sources']:
        path = A / source['path']
        assert sha(path) == source['sha256']
        sources.append({**source, 'path': str(path)})
    manifest = A / 'manifest.json'
    write(manifest, {'schema': MANIFEST_VERSION, 'sources': sources})
    bundle = F / 'reproduce/source'
    old = (bundle / 'modular_sources.original.py').read_bytes()
    assert sha(ROOT / 'data/modular_sources.py') == sha(bundle / 'modular_sources.candidate.py')
    assert b'for index in range(len(self)):\n            yield self[index]' in old
    shutil.copyfile(__file__, bundle / 'benchmark.py')
    shutil.copyfile(F / 'code/capture.py', bundle / 'capture.py')
    invocation = dict(
        argv=sys.argv,
        agent_model='Codex / GPT-6',
        python=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
        gpu=None,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], text=True),
        manifest=str(manifest),
        manifest_sha256=sha(manifest),
        sources=capture['sources'],
        captured_train_rows=15000,
        source_hashes={p.name: sha(p) for p in bundle.iterdir() if p.is_file()},
        pretraining='none',
        training='none',
        loss='none',
        synthetic_data='none',
        validation_test='not used: no predictive evaluation',
        cache='warm, after full content parity pass; no rows cached by the reader',
        repeats=3,
        order='baseline/stream; stream/baseline; baseline/stream',
        non_default_environment_overrides={},
        started_unix=time.time(),
    )
    write(F / 'reproduce/launch.json', invocation)
    start = time.perf_counter()
    index = ObservationIndex(manifest)
    construction_seconds = time.perf_counter() - start
    assert len(index) == 15000
    count = 0
    for left, right in zip(original(index), iter(index), strict=True):
        assert left.to_dict() == right.to_dict()
        count += 1
    assert count == len(index)
    write(
        F / 'results/content-verification.json',
        dict(
            status='passed',
            rows=count,
            compared='every normalized observation field and order',
            manifest_sha256=sha(manifest),
            source_hashes=invocation['source_hashes'],
        ),
    )
    rows = []
    expected_digest = None
    for repeat in range(3):
        conditions = [('original', original), ('sequential', iter)]
        if repeat % 2:
            conditions.reverse()
        for name, factory in conditions:
            gc.collect()
            digest, count = hashlib.sha256(), 0
            start = time.perf_counter()
            for row in factory(index):
                digest.update(row.observation_id.encode() + b'\n')
                count += 1
            seconds = time.perf_counter() - start
            if expected_digest is None:
                expected_digest = digest.hexdigest()
            assert count == len(index) and digest.hexdigest() == expected_digest
            result = dict(
                condition=name,
                repeat=repeat,
                rows=count,
                seconds=seconds,
                rows_per_second=count / seconds,
                identity_digest=digest.hexdigest(),
            )
            rows.append(result)
            print(name, repeat, round(seconds, 3), 'seconds', flush=True)
    medians = {
        name: statistics.median(r['seconds'] for r in rows if r['condition'] == name)
        for name in ('original', 'sequential')
    }
    speedup = medians['original'] / medians['sequential']
    summary = dict(
        status='completed',
        results=rows,
        median_seconds=medians,
        median_speedup=speedup,
        throughput_gate_passed=speedup >= 1.10,
        construction_seconds=construction_seconds,
        exact_content_verified=count,
        maximum_resident_set_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_unit='bytes' if sys.platform == 'darwin' else 'KiB',
        finished_unix=time.time(),
        interpretation='Local warm-cache TRAIN-prefix reader benchmark only; not an estimate of full H100 fit speedup',
    )
    write(F / 'results/summary.json', summary)
    with (F / 'results/timings.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Median speedup', round(speedup, 4), 'gate', summary['throughput_gate_passed'])


if __name__ == '__main__':
    main()
