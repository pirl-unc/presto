"""Bounded source compilation; this launcher does not train a model."""

import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
PYTHON = Path('/Users/iskander/code/shared-virtual-env/bin/python')
HITLIST = Path('/private/tmp/hitlist-mapp2025-20261024')
SOURCE = REPOSITORY / 'data/mapp2025_source/2026-10-24_2130'
PACKAGE = REPOSITORY / 'data/mapp2025_source/2026-10-24_2210_canonical'


def main():
    if '--worker' in sys.argv:
        from hitlist.goldberg_mapp import write_goldberg_mapp_package

        start = time.monotonic()
        manifest = write_goldberg_mapp_package(SOURCE, PACKAGE)
        print(json.dumps({'manifest': str(manifest), 'wall_seconds': time.monotonic() - start,
                          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        return
    receipt = EXPERIMENT / 'results/build-resource.json'
    if receipt.exists():
        raise FileExistsError('Preserve the prior attempt; register a new attempt before rerunning')
    command = [str(PYTHON), str(Path(__file__).resolve()), '--worker']
    env = dict(os.environ, PYTHONPATH=str(HITLIST), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    start, peak, reason = time.monotonic(), 0, None
    with (EXPERIMENT / 'launch_logs/build.log').open('w') as log:
        process = subprocess.Popen(command, cwd=HITLIST, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        probe = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, probe.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak > 6 * 1024**3:
                reason = 'process_memory_cap'
            elif time.monotonic() - start > 1200:
                reason = 'compile_and_internal_replay_time_cap'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                break
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        code = process.wait(timeout=30)
    size = sum(p.stat().st_size for p in PACKAGE.rglob('*') if p.is_file()) if PACKAGE.exists() else 0
    audit_size = sum(p.stat().st_size for p in EXPERIMENT.rglob('*') if p.is_file())
    result = dict(command=command, exit_code=code, wall_seconds=time.monotonic() - start,
                  peak_polled_rss_bytes=peak, package_size_bytes=size, audit_size_bytes=audit_size,
                  termination_reason=reason, resource_gate=reason is None and size + audit_size <= 10 * 1024**3)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if code or not result['resource_gate']:
        raise SystemExit(code or 1)
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    (EXPERIMENT / 'results/package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
