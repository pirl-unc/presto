"""Immutable source pins and bounded metadata-only scan execution."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

import psutil

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
BASE = Path(
    '/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z'
)
SOURCES = {
    'binding.parquet': {
        'sha256': '2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0',
        'rows': 814268,
    },
    'other_assays.parquet': {
        'sha256': 'c40f5227ded408830885ecba8336dbdd701172df9573b0d32e5e8a5f8fc947c2',
        'rows': 26537,
    },
}
FIELDS = [
    'source',
    'response_measured',
    'assay_method',
    'measurement_units',
    'measurement_inequality',
    'mhc_class',
    'has_ptm',
    'is_binding_assay',
    'is_ms_observation',
]
EXTRA = ['quantitative_value', 'assay_iri', 'reference_iri']
GIB = 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


@contextmanager
def guarded(action):
    path = FAMILY / 'reproduce' / f'{action}-{time.time_ns()}.json'
    started = time.monotonic()
    process = psutil.Process()
    state = dict(
        status='running',
        agent_model='Codex / GPT-6',
        utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv,
        cwd=str(ROOT),
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        device='cpu',
        requested_gpu=None,
        observed_gpu=None,
        packages={n: version(n) for n in ('pyarrow', 'psutil')},
        git_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        environment={
            k: os.environ.get(k)
            for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONPATH')
        },
        source_pins=SOURCES,
        code_hashes={
            str(p.relative_to(FAMILY)): sha(p) for p in sorted((FAMILY / 'code').glob('*.py'))
        },
        limits={
            'seconds': 300,
            'rss_bytes': 3 * GIB,
            'startup_ram_bytes': 10 * GIB,
            'ram_floor_bytes': 5 * GIB,
            'startup_disk_bytes': 15 * GIB,
            'disk_floor_bytes': 10 * GIB,
        },
        startup_ram_bytes=psutil.virtual_memory().available,
        startup_disk_bytes=shutil.disk_usage(ROOT).free,
    )
    write(path, state)
    stop = threading.Event()
    peak = [process.memory_info().rss]
    violation = [None]

    def fail(_sig, _frame):
        raise RuntimeError('resource guard: ' + str(violation[0]))

    def monitor():
        while not stop.wait(0.25):
            try:
                rss = process.memory_info().rss
                peak[0] = max(peak[0], rss)
                if rss > 3 * GIB:
                    violation[0] = 'rss'
                elif psutil.virtual_memory().available < 5 * GIB:
                    violation[0] = 'available_ram'
                elif shutil.disk_usage(ROOT).free < 10 * GIB:
                    violation[0] = 'disk'
                elif time.monotonic() - started > 300:
                    violation[0] = 'time'
            except Exception as exc:
                violation[0] = repr(exc)
            if violation[0]:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    old = signal.signal(signal.SIGUSR1, fail)
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        if state['startup_ram_bytes'] < 10 * GIB or state['startup_disk_bytes'] < 15 * GIB:
            raise RuntimeError('insufficient startup resources')
        thread.start()
        yield state
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', failure=repr(exc))
        raise
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join()
        signal.signal(signal.SIGUSR1, old)
        state.update(
            elapsed_seconds=time.monotonic() - started,
            peak_polled_rss_bytes=peak[0],
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource_violation=violation[0],
        )
        write(path, state)
