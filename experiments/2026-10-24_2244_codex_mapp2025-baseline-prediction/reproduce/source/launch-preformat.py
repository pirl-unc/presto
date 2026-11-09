"""Run one frozen stage with resource bounds and immutable attempt receipts."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main():
    stage = sys.argv[1]
    if stage not in ('admit', 'audit', 'qualify', 'fit', 'evaluate', 'replay'):
        raise ValueError(stage)
    receipt = ROOT / f'results/{stage}-resource.json'
    if receipt.exists():
        raise FileExistsError('Preserve prior attempts; no automatic retry')
    config = json.loads((ROOT / 'reproduce/launch.json').read_text())
    script = ROOT / ('analysis' if stage in ('audit', 'replay') else 'code') / f'{stage}.py'
    snapshot = ROOT / f'reproduce/source/{stage}.py'
    snapshot.write_bytes(script.read_bytes())
    for name in ('core.py',):
        shared = ROOT / 'code' / name
        target = ROOT / 'reproduce/source' / name
        if shared.exists():
            if target.exists():
                assert target.read_bytes() == shared.read_bytes(), 'Core changed after first stage'
            else:
                target.write_bytes(shared.read_bytes())
    command = [config['python'], str(script)]
    start, peak, reason = time.monotonic(), 0, None
    with (ROOT / f'launch_logs/{stage}.log').open('w') as output:
        process = subprocess.Popen(command, cwd=REPO, env={**os.environ, **config['environment_overrides']},
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        probe = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, probe.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak > 6*1024**3:
                reason = 'rss_limit'
            elif time.monotonic() - start > 1200:
                reason = 'stage_time_limit'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                break
            try:
                process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
        code = process.wait(timeout=30)
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    package = REPO / config['recipe_directory']
    size += sum(p.stat().st_size for p in package.rglob('*') if p.is_file()) if package.exists() else 0
    result = {'stage':stage, 'command':command, 'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
              'wall_seconds':time.monotonic()-start, 'peak_polled_rss_bytes':peak,
              'exit_code':code, 'termination_reason':reason, 'output_bytes':size,
              'resource_gate':reason is None and size <= 10*1024**3}
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if code or not result['resource_gate']:
        raise SystemExit(code or 1)

if __name__ == '__main__':
    main()
